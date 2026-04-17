from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer


def load_config(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def format_chat_example(example: dict, tokenizer) -> dict:
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def detect_gpu_memory_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    props = torch.cuda.get_device_properties(0)
    return round(props.total_memory / (1024**3), 2)


def is_bf16_available() -> bool:
    return torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()


def auto_adjust_config(config: dict) -> dict:
    effective = dict(config)
    gpu_memory_gb = detect_gpu_memory_gb()
    effective["detected_gpu_memory_gb"] = gpu_memory_gb
    effective["use_bf16"] = bool(config.get("use_bf16", False) and is_bf16_available())
    effective["use_fp16"] = bool(torch.cuda.is_available() and not effective["use_bf16"])

    if gpu_memory_gb and gpu_memory_gb <= 16:
        effective["max_seq_length"] = min(int(effective.get("max_seq_length", 2048)), 1024)
        effective["per_device_train_batch_size"] = 1
        effective["gradient_accumulation_steps"] = max(int(effective.get("gradient_accumulation_steps", 4)), 8)
        effective["lora_r"] = min(int(effective.get("lora_r", 16)), 8)
        effective["logging_steps"] = min(int(effective.get("logging_steps", 10)), 5)
        effective["auto_adjust_reason"] = "Applied low-VRAM profile (<=16GB GPU)."
    elif gpu_memory_gb and gpu_memory_gb <= 24:
        effective["max_seq_length"] = min(int(effective.get("max_seq_length", 2048)), 1536)
        effective["per_device_train_batch_size"] = min(int(effective.get("per_device_train_batch_size", 2)), 1)
        effective["gradient_accumulation_steps"] = max(int(effective.get("gradient_accumulation_steps", 4)), 6)
        effective["auto_adjust_reason"] = "Applied medium-VRAM profile (<=24GB GPU)."
    else:
        effective["auto_adjust_reason"] = "No VRAM downscaling applied."

    effective["gradient_checkpointing"] = True
    return effective


def print_effective_config(config: dict) -> None:
    printable = {
        "model_name": config["model_name"],
        "max_seq_length": config["max_seq_length"],
        "load_in_4bit": config.get("load_in_4bit", True),
        "per_device_train_batch_size": config["per_device_train_batch_size"],
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "learning_rate": config["learning_rate"],
        "lora_r": config["lora_r"],
        "detected_gpu_memory_gb": config.get("detected_gpu_memory_gb", 0.0),
        "use_fp16": config.get("use_fp16", False),
        "use_bf16": config.get("use_bf16", False),
        "auto_adjust_reason": config.get("auto_adjust_reason", ""),
    }
    print(json.dumps(printable, indent=2))


def build_sft_trainer(
    model,
    tokenizer,
    train_dataset,
    peft_config,
    training_args,
    max_seq_length: int,
):
    signature = inspect.signature(SFTTrainer.__init__)
    parameters = signature.parameters

    trainer_kwargs = {
        "model": model,
        "train_dataset": train_dataset,
        "peft_config": peft_config,
        "args": training_args,
    }

    if "tokenizer" in parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in parameters:
        trainer_kwargs["processing_class"] = tokenizer

    if "dataset_text_field" in parameters:
        trainer_kwargs["dataset_text_field"] = "text"

    if "max_seq_length" in parameters:
        trainer_kwargs["max_seq_length"] = max_seq_length

    return SFTTrainer(**trainer_kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SFT model with LoRA/QLoRA.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.model_name:
        config["model_name"] = args.model_name
    config = auto_adjust_config(config)
    model_name = config["model_name"]
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print_effective_config(config)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = config["max_seq_length"]

    quantization_config = None
    compute_dtype = torch.bfloat16 if config.get("use_bf16", False) else torch.float16
    if config.get("load_in_4bit", True):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        torch_dtype=compute_dtype if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    peft_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config["target_modules"],
    )

    dataset = load_dataset("json", data_files=args.dataset_file, split="train")
    dataset = dataset.map(lambda row: format_chat_example(row, tokenizer), remove_columns=dataset.column_names)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        warmup_steps=max(1, int((len(dataset) / max(config["per_device_train_batch_size"], 1)) * config["num_train_epochs"] * 0.03)),
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        fp16=config.get("use_fp16", False),
        bf16=config.get("use_bf16", False),
        report_to="none",
        optim="paged_adamw_8bit" if config.get("load_in_4bit", True) else "adamw_torch",
        lr_scheduler_type="cosine",
        gradient_checkpointing=config.get("gradient_checkpointing", True),
    )

    trainer = build_sft_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        peft_config=peft_config,
        training_args=training_args,
        max_seq_length=config["max_seq_length"],
    )
    try:
        trainer.train()
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise RuntimeError(
                "CUDA out of memory during training. "
                "Try a smaller base model with --model-name, or reduce max_seq_length in configs/sft_config.json."
            ) from exc
        raise
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved adapter and tokenizer to {args.output_dir}")


if __name__ == "__main__":
    main()
