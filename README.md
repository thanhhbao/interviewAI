# InterviewAI Colab Training Skeleton

Bo skeleton nay tap trung vao 2 lop:

1. Train pipeline tren Colab cho cac task LLM text.
2. Runtime pipeline da modal gom:
   - resume parsing
   - question generation
   - audio analysis voi Whisper
   - vision analysis voi MediaPipe
   - fusion scoring
   - report JSON/PDF

## Cau truc

```text
.
├── configs/
│   └── sft_config.json
├── data/
│   └── sample/
│       ├── answers/
│       ├── jds/
│       └── resumes/
├── scripts/
│   ├── prepare_dataset.py
│   ├── prepare_kaggle_multitask.py
│   ├── run_multimodal_session.py
│   ├── run_demo_pipeline.py
│   └── train_sft.py
├── src/
│   └── interview_ai/
│       ├── __init__.py
│       ├── augmentation.py
│       ├── audio.py
│       ├── dataset_adapters.py
│       ├── fusion.py
│       ├── io.py
│       ├── parsers.py
│       ├── pipeline.py
│       ├── prompts.py
│       ├── report.py
│       ├── schemas.py
│       ├── session.py
│       └── scoring.py
│       └── vision.py
└── requirements-colab.txt
```

## Luong train tren Colab

### 1. Cai dependencies

```bash
pip install -r requirements-colab.txt
```

### 2. Dua du lieu vao `data/`

- `data/sample/resumes/`: CV dang `.pdf`, `.docx`, `.txt`
- `data/sample/jds/`: Job Description dang `.txt`, `.md`, `.json`
- `data/sample/answers/`: file `.jsonl` chua transcript + rubric cho bai train danh gia cau tra loi

### 3. Tao dataset instruction tuning

```bash
python scripts/prepare_dataset.py \
  --resume-dir data/sample/resumes \
  --jd-dir data/sample/jds \
  --answer-file data/sample/answers/train_answers.jsonl \
  --output-file output/train_sft.jsonl
```

Hoac neu ban da download 2 dataset Kaggle ve local/Drive:

```bash
python scripts/prepare_kaggle_multitask.py \
  --resume-dataset-dir /content/datasets/resume-entities-for-ner \
  --question-dataset-dir /content/datasets/software-engineering-interview-questions-dataset \
  --augment \
  --num-augments 2 \
  --output-file output/train_sft.jsonl
```

### 4. Train LoRA/QLoRA

```bash
python scripts/train_sft.py \
  --config configs/sft_config.json \
  --dataset-file output/train_sft.jsonl \
  --output-dir output/qwen-resume-lora
```

### 5. Chay demo inference

```bash
python scripts/run_demo_pipeline.py \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --adapter-dir output/qwen-resume-lora \
  --resume-file data/sample/resumes/sample_resume.txt \
  --jd-file data/sample/jds/sample_jd.txt
```

### 6. Chay full multimodal session

```bash
python scripts/run_multimodal_session.py \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --resume-file data/sample/resumes/sample_resume.txt \
  --jd-file data/sample/jds/sample_jd.txt \
  --answer-audio data/sample/audio/sample_answer.txt \
  --vision-file data/sample/vision/sample_vision_frames.json \
  --report-dir output/report
```

## Dinh dang dataset train

Script `prepare_dataset.py` sinh ra JSONL theo format chat:

```json
{
  "messages": [
    {"role": "system", "content": "You are an interview AI assistant."},
    {"role": "user", "content": "Extract the following resume into JSON..."},
    {"role": "assistant", "content": "{\"candidate_profile\": ... }"}
  ],
  "task": "resume_extract"
}
```

## Ghi chu quan trong

- Skeleton nay tao `weak labels` cho task parse CV, optimize CV, generate questions.
- `prepare_kaggle_multitask.py` da noi truc tiep 2 dataset Kaggle vao train pipeline:
  - `Resume Entities for NER` -> task `resume_extract`
  - `Software Engineering Interview Questions Dataset` -> task `question_generation`
- `augmentation.py` bo sung offline data augmentation:
  - bien doi format CV nhung giu nguyen target JSON
  - tao nhieu prompt/JD bien the cho task question generation
- Task `answer_evaluation` se tot hon neu ban co du lieu gan nhan that.
- Neu Colab dung GPU T4, nen giu model base nho (`1.5B` hoac `3B`) va train bang QLoRA.
- `audio.py` ho tro 2 che do:
  - co `whisper`: transcribe audio that
  - khong co `whisper`: doc transcript tu file `.txt` de demo pipeline
- `vision.py` ho tro 2 che do:
  - co `mediapipe`: phan tich frame/image that
  - khong co `mediapipe`: doc file JSON feature da trich san de demo
- `session.py` la orchestration layer de ghep CV + LLM + audio + vision + fusion + report.

## Anh xa voi pipeline ban dau

1. `Resume Processing`
   - `parsers.py`
   - `prepare_dataset.py`
   - `pipeline.py`
2. `Interview Setup`
   - `pipeline.py`
   - `session.py`
3. `Real-time Analysis`
   - `audio.py`
   - `vision.py`
4. `Evaluation & Feedback`
   - `scoring.py`
   - `fusion.py`
   - `report.py`
   - `session.py`

## Nguon dataset text

- Resume Entities for NER: https://www.kaggle.com/datasets/dataturks/resume-entities-for-ner
- Software Engineering Interview Questions Dataset: https://www.kaggle.com/datasets/syedmharis/software-engineering-interview-questions-dataset
