# InterviewAI

InterviewAI is a multimodal interview-assistant skeleton for:

- resume parsing
- job-description matching
- interview question generation
- speech analysis
- vision-based behavior analysis
- score fusion
- report generation

The repository is designed around two layers:

1. A `training pipeline` for fine-tuning a text LLM on Colab.
2. A `runtime multimodal pipeline` that combines LLM, Whisper, MediaPipe, fusion logic, and reporting.

## What Is Already Implemented

### Training

- LoRA / QLoRA fine-tuning for one multitask LLM
- dataset preparation from local samples
- dataset preparation directly from the two Kaggle datasets you selected
- offline data augmentation for resume extraction and question generation

### Runtime

- resume parsing and resume optimization
- question generation
- audio analysis with Whisper or transcript fallback
- vision analysis with MediaPipe-ready hooks or precomputed frame metrics
- score fusion
- JSON and PDF report generation

## Project Structure

```text
.
├── configs/
│   └── sft_config.json
├── data/
│   └── sample/
│       ├── answers/
│       ├── audio/
│       ├── jds/
│       ├── resumes/
│       └── vision/
├── scripts/
│   ├── prepare_dataset.py
│   ├── prepare_kaggle_multitask.py
│   ├── prepare_round2_dataset.py
│   ├── test_model.py
│   ├── run_demo_pipeline.py
│   ├── run_turn_based_session.py
│   ├── run_multimodal_session.py
│   └── train_sft.py
├── src/
│   └── interview_ai/
│       ├── __init__.py
│       ├── augmentation.py
│       ├── audio.py
│       ├── conversation.py
│       ├── dataset_adapters.py
│       ├── fusion.py
│       ├── io.py
│       ├── parsers.py
│       ├── pipeline.py
│       ├── prompts.py
│       ├── report.py
│       ├── schemas.py
│       ├── scoring.py
│       ├── session.py
│       ├── tts.py
│       └── vision.py
└── requirements-colab.txt
```

## Pipeline Mapping

The codebase matches the original system design as follows:

### 1. Resume Processing

- `src/interview_ai/parsers.py`
- `src/interview_ai/pipeline.py`
- `scripts/prepare_dataset.py`
- `scripts/prepare_kaggle_multitask.py`

### 2. Interview Setup

- `src/interview_ai/pipeline.py`
- `src/interview_ai/session.py`
- `src/interview_ai/conversation.py`
- `src/interview_ai/tts.py`

### 3. Real-Time Analysis

- `src/interview_ai/audio.py`
- `src/interview_ai/vision.py`

### 4. Evaluation and Feedback

- `src/interview_ai/scoring.py`
- `src/interview_ai/fusion.py`
- `src/interview_ai/report.py`
- `src/interview_ai/session.py`

## Datasets

This repository is currently set up to train one multitask text model from these two datasets:

- Resume Entities for NER  
  https://www.kaggle.com/datasets/dataturks/resume-entities-for-ner
- Software Engineering Interview Questions Dataset  
  https://www.kaggle.com/datasets/syedmharis/software-engineering-interview-questions-dataset

### Verified Raw Dataset Schema

The adapters were updated against the actual zip files:

- `resume-entities dataset.zip`
  - contains `Entity Recognition in Resumes.json`
  - stored as `JSONL`
  - fields include `content`, `annotation`, `extras`
- `interview_questions dataset.zip`
  - contains `Software Questions.csv`
  - encoded as `cp1252`
  - columns are:
    - `Question Number`
    - `Question`
    - `Answer`
    - `Category`
    - `Difficulty`

## Round 2 Dataset Strategy

For a more realistic Vietnamese interview flow, the repository now supports generating a synthetic Vietnamese behavior dataset on top of the original question dataset.

This round-2 dataset includes:

- `question_generation_vi`
- `answer_evaluation_vi`
- `follow_up_vi`
- `next_action_policy_vi`
- `resume_extract`
- `question_generation`

Prepare it with:

```bash
python scripts/prepare_round2_dataset.py \
  --resume-dataset-dir "/content/resume-entities dataset.zip" \
  --question-dataset-dir "/content/interview_questions dataset.zip" \
  --augment \
  --num-augments 2 \
  --output-file output/train_round2.jsonl
```

Then continue training from the round-1 adapter:

```bash
python scripts/train_sft.py \
  --config configs/sft_config.json \
  --dataset-file output/train_round2.jsonl \
  --resume-adapter-dir output/qwen-resume-lora \
  --output-dir output/qwen-resume-lora-round2 \
  --num-train-epochs 1
```

## Training Strategy

This repository is set up for `one multitask text model`, not two separate models.

The current multitask training targets:

- `resume_extract`
- `question_generation`

Additional tasks such as `resume_optimize` and `answer_evaluation` are supported in the codebase, but they are not fully backed by the two Kaggle datasets alone. They will benefit from extra labeled data later.

## Offline Data Augmentation

`src/interview_ai/augmentation.py` adds offline augmentation before training.

### Resume augmentation

- section header variation
- phone-number formatting variation
- section/block reordering
- spacing normalization

The target JSON remains unchanged.

### Question-generation augmentation

- multiple prompt templates
- multiple synthetic JD formulations
- multiple instruction styles for the same target question

## Colab Setup

Install dependencies:

```bash
pip install -r requirements-colab.txt
```

## Prepare Training Data from the Kaggle Zip Files

You can point the script directly to the zip files. Manual extraction is not required.

```bash
python scripts/prepare_kaggle_multitask.py \
  --resume-dataset-dir "/content/resume-entities dataset.zip" \
  --question-dataset-dir "/content/interview_questions dataset.zip" \
  --augment \
  --num-augments 2 \
  --output-file output/train_sft.jsonl
```

## Preview the Generated Training File

Recommended before long training runs:

```bash
wc -l output/train_sft.jsonl
head -n 3 output/train_sft.jsonl
```

## Train the Model

```bash
python scripts/train_sft.py \
  --config configs/sft_config.json \
  --dataset-file output/train_sft.jsonl \
  --output-dir output/qwen-resume-lora
```

To continue training from an existing LoRA adapter:

```bash
python scripts/train_sft.py \
  --config configs/sft_config.json \
  --dataset-file output/train_sft.jsonl \
  --resume-adapter-dir output/qwen-resume-lora \
  --output-dir output/qwen-resume-lora-round2 \
  --num-train-epochs 1
```

## Run Text-Only Demo Inference

```bash
python scripts/run_demo_pipeline.py \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --adapter-dir output/qwen-resume-lora \
  --resume-file data/sample/resumes/sample_resume.txt \
  --jd-file data/sample/jds/sample_jd.txt
```

## Quick Model Test After Each Round

```bash
python scripts/test_model.py \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --adapter-dir output/qwen-resume-lora \
  --resume-file data/sample/resumes/sample_resume.txt \
  --jd-file data/sample/jds/sample_jd.txt \
  --output-file output/test_round1.json
```

This script checks:

- whether `resume_extract` returns valid JSON
- whether `question_generation` returns valid JSON
- whether question generation returns a non-empty list

## Run Full Multimodal Session

```bash
python scripts/run_multimodal_session.py \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --resume-file data/sample/resumes/sample_resume.txt \
  --jd-file data/sample/jds/sample_jd.txt \
  --answer-audio data/sample/audio/sample_answer.txt \
  --vision-file data/sample/vision/sample_vision_frames.json \
  --report-dir output/report
```

## Run Turn-Based Push-to-Talk Interview Flow

This flow is intended for the simpler and more stable interview UX:

1. the system shows or speaks one question
2. the user presses the mic button and records one answer
3. the answer audio is transcribed with Whisper
4. the LLM evaluates the answer
5. the system decides whether to ask a follow-up question or move to the next planned question

Example:

```bash
python scripts/run_turn_based_session.py \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --adapter-dir output/qwen-resume-lora \
  --resume-file data/sample/resumes/sample_resume.txt \
  --jd-file data/sample/jds/sample_jd.txt \
  --answer-audio data/sample/audio/sample_answer.txt data/sample/audio/sample_answer.txt \
  --session-output output/conversation/session.json \
  --tts-output-dir output/conversation/tts
```

Current implementation note:

- `src/interview_ai/tts.py` is a local stub that writes the spoken text to a file
- replace it later with a real local TTS engine such as Piper or Coqui TTS

## Training Data Format

All training data is converted to a chat-style JSONL format:

```json
{
  "messages": [
    { "role": "system", "content": "You are an interview AI assistant." },
    { "role": "user", "content": "Extract the following resume into JSON..." },
    { "role": "assistant", "content": "{\"candidate_profile\": ... }" }
  ],
  "task": "resume_extract",
  "meta": {
    "dataset": "resume_entities_for_ner"
  }
}
```

## Notes

- The LLM is the part that is fine-tuned.
- Whisper, MediaPipe, and TTS-related components are intended to stay pretrained or rule-based for now.
- The runtime multimodal pipeline is implemented as a skeleton suitable for research demos and further extension.
- For Colab T4, a small base model such as `Qwen2.5-1.5B-Instruct` is the practical default.

## Recommended Next Steps

If you want stronger model quality, the next additions should be:

1. train/validation split and dataset deduplication
2. synthetic or manually labeled `answer_evaluation` data
3. real MediaPipe landmark extraction instead of placeholder image-level hooks
4. a Colab notebook for end-to-end execution
