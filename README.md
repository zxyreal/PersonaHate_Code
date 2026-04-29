# PersonaHate

A persona-driven pipeline for generating diverse, labeled hate speech data to train and evaluate hate speech detectors.

## Pipeline Overview

```
text_2_persona/          →  persona_2_speech/         →  train/
Persona Construction        Speech Generation &          Model Training &
& Selection                 Annotation                   Evaluation
```

## Directory Structure

### `text_2_persona/` — Persona Construction & Selection

| File | Description |
|------|-------------|
| `extract_persona_from_4chan.py` | Extract persona descriptions from 4chan /pol/ posts using GPT-4o |
| `process_personas.py` | Embed personas, deduplicate (cosine similarity > 0.9), merge pools |
| `da_fps.py` | Core FPS / DA-FPS / Random sampling algorithms |
| `run_fps_comparison.py` | Compare selection methods, generate UMAP visualizations |
| `process_personas_fast.py` | Fast variant using Faiss-accelerated deduplication |
| `run_dafps_personahub.py` | Process and select PersonaHub personas |

**Data:**
- `data/persona_merged.jsonl` — Merged persona pool (457,473: 257K 4chan + 200K PersonaHub)
- `data/persona_selected_2285_fps_stratified.jsonl` — FPS-selected personas (2,285)

### `persona_2_speech/` — Speech Generation & Annotation

| File | Description |
|------|-------------|
| `generate_hatespeech.py` | Unified generation script (8 models: API + vLLM) |
| `model_adapters.py` | Model-specific adapters for speech extraction and refusal detection |
| `identity.py` | 34 identity groups across 6 categories |
| `vote_label_6judge.py` | 6-judge voting annotation (GPT-4o-mini, Claude, Gemini, Gemma, OpenAI Moderation, LlamaGuard) |
| `postprocess.py` | Post-processing: refusal detection, normalization, length filtering |
| `newwave_*.py` | NewWave extension for emerging hate topics |

**Data:**
- `hatespeech/personahate_train_balanced_6judge.jsonl` — Group-balanced training set (67,452 samples)
- `hatespeech/{model}/` — Per-model generated speech with 6-judge labels (791,283 total)

### `train/` — Model Training & Evaluation

| File | Description |
|------|-------------|
| `train_encoders.py` | Train encoder models (BERT, RoBERTa, DeBERTa-v3, XLM-RoBERTa) |
| `train_llm_sft.py` | SFT fine-tune Llama-3 / LlamaGuard on PersonaHate |
| `train_cardiff.py` | Fine-tune Cardiff-RoBERTa on PersonaHate |
| `eval_encoder.py` | Evaluate encoder models on hate speech benchmarks |
| `eval_llm_sft.py` | Evaluate LLM SFT models via vLLM |
| `eval_pergroup.py` | Per-identity-group evaluation on HateBenchSet |
| `eval_baselines.py` | Evaluate commercial APIs (GPT-4o-mini, Claude, Gemini, etc.) |
| `eval_pretrained_baselines.py` | Evaluate off-the-shelf hate speech models |
| `ablation_*.py` | Ablation experiments (scaling, diversity, group coverage) |
| `newwave_*.py` | NewWave pipeline: generate, vote, train, evaluate |
| `plot_*.py` | Visualization scripts (t-SNE, UMAP, scaling curves) |

## Dataset Statistics

| Property | Value |
|----------|-------|
| Total generated samples | 791,283 |
| Generator models | 8 (3 API + 5 open-weight) |
| Identity groups | 34 (6 categories) |
| Judge models | 6 |
| Voting threshold | ≥ 4/6 |
| Training set (balanced) | 67,452 (33,726 hate / 33,726 non-hate) |
| Unique personas | 2,285 (1,000 PersonaHub + 1,285 4chan) |

## Requirements

```
torch
transformers
datasets
sentence-transformers
sklearn
openai
anthropic
google-generativeai
vllm
faiss-gpu
detoxify
```

## Usage

### 1. Persona Construction
```bash
# Extract personas from 4chan /pol/ dataset
python text_2_persona/extract_persona_from_4chan.py --input pol_data.ndjson --output personas.jsonl

# Deduplicate and merge
python text_2_persona/process_personas.py --input personas.jsonl --threshold 0.9

# Select diverse subset via FPS
python text_2_persona/run_fps_comparison.py
```

### 2. Speech Generation & Annotation
```bash
# Generate speech from personas
python persona_2_speech/generate_hatespeech.py --model gpt4o-mini --personas data/persona_selected_2285_fps_stratified.jsonl

# Annotate with 6-judge voting
python persona_2_speech/vote_label_6judge.py --input generated_speech.jsonl
```

### 3. Training & Evaluation
```bash
# Train encoder
python train/train_encoders.py --model deberta-v3-base

# Evaluate
python train/eval_encoder.py --model-path models/deberta-v3-base_personahate/final
```
