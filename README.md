# PersonaHate

A persona-driven pipeline for generating diverse, labeled hate speech data to train and evaluate hate speech detectors.

## Requirements

- Python 3.8+
- OpenAI API key
- Hugging Face token (for downloading models)
- Google Cloud project (for Vertex AI: Gemini, Claude)

## Installation

```bash
pip install -r requirements.txt
```

## Data

Download the following datasets and place them in `text_2_persona/data/`:

| File | Description | Source |
|------|-------------|--------|
| `pol_062016-112019_labeled.ndjson` | 4chan /pol/ dataset with toxicity labels | [TODO: Add link] |
| `persona_personahub_200k.jsonl` | PersonaHub 200k personas | [TODO: Add link] |

**Pre-extracted personas (from our paper):**
- `text_2_persona/data/output/personas_paper.jsonl` - 2,285 diverse personas selected via FPS (1,285 from 4chan + 1,000 from PersonaHub)

**Pre-built training subset (for quick testing):**
- `train/data/personahate_train_balanced_6judge_subset.jsonl` - 13,600 samples (200 hate + 200 non-hate per group)

**Evaluation benchmarks** (place in `train/data/benchmarks/`, all available on HuggingFace):

For multi-class datasets, we focus on binary hate speech detection by retaining hate and non-hate samples and removing samples labeled as offensive.

| File | Dataset |
|------|---------|
| `all_7_models_voting.csv` | HateXplain |
| `davidson_labeled_data.csv` | Davidson |
| `mhs_aggregated.csv` | Measuring Hate Speech |
| (auto-downloaded) | HateBenchSet |
| `new_wave_hate/data_ground_truth.csv` | NewWave |

## Configuration

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_api_key_here
GOOGLE_CLOUD_PROJECT=your_gcp_project_id
GOOGLE_CLOUD_LOCATION=us-central1
HF_TOKEN=your_huggingface_token_here
```

## Usage

### Step 1: Extract Personas

Extract persona descriptions from toxic posts using GPT-4o.

```bash
cd text_2_persona
python extract_persona_from_4chan.py
```

| Argument | Description |
|----------|-------------|
| `--input_file` | Input NDJSON file path |
| `--output_path` | Output file path |
| `--sample_size` | Number of posts to process |
| `--num_workers` | Number of parallel workers (default: CPU count - 1) |

### Step 2: Process Personas

Compute embeddings, remove duplicates, and select diverse personas using FPS.

**For 4chan personas:**
```bash
cd text_2_persona
python process_personas_4chan.py
```

**For PersonaHub personas:**
```bash
cd text_2_persona
python process_personas_personahub.py
```

| Argument | Description |
|----------|-------------|
| `--input` | Input JSONL file |
| `--threshold` | Similarity threshold for deduplication (default: 0.9) |
| `--n_select` | Number of diverse personas to select |
| `--skip_embedding` | Skip embedding computation, use cached embeddings |

Output files:
- `persona_deduplicated.jsonl` - Personas after removing duplicates
- `persona_selected_{n}.jsonl` - Final diverse persona selection

### Step 3: Merge Personas

Merge 4chan and PersonaHub personas into a single file.

```bash
cd text_2_persona
python merge_personas.py
```

Output: `personas_merged.jsonl`

### Step 4: Generate Speech

Generate speech from personas using various LLMs.

```bash
cd persona_2_speech
python generate_hatespeech.py --model gpt4o-mini
```

Available models:
- `gpt4o-mini` - GPT-4o-mini (OpenAI)
- `gemini` - Gemini 2.5 Flash (Vertex AI)
- `claude` - Claude 3 Haiku (Vertex AI)
- `llama-3.1-8b`, `qwen2.5-7b`, `deepseek-r1`, `mistral-7b`, `gemma-2-9b` - Local models (vLLM)

**For vLLM models**, start the server first:
```bash
python -m vllm.entrypoints.openai.api_server --model <model_name>
```

| Argument | Description |
|----------|-------------|
| `--model` | Model to use |
| `--persona-file` | Specific persona file |
| `--max-workers` | Number of parallel workers |
| `--num-personas` | Number of personas to use |

### Step 5: Postprocess

Clean outputs and fix missed refusals.

```bash
cd persona_2_speech
python postprocess.py
```

### Step 6: Label with Judges

Label generated speech using 6 judges (majority vote).

```bash
cd persona_2_speech
python vote_label_6judge.py --model gpt4o-mini
```

Judges: GPT-4o-mini, Claude-3-Haiku, Gemini-2.5-Flash, Gemma-2-9B, OpenAI Moderation, LlamaGuard-3

| Argument | Description |
|----------|-------------|
| `--model` | Specific generator model to label |
| `--workers` | Number of parallel workers |
| `--api-only` | Only use API judges (skip vLLM) |
| `--phase1` | Use 5 judges (no LlamaGuard) |
| `--phase2` | Add LlamaGuard to existing labels |

### Step 7: Build Training Set

Build group-balanced training set from labeled data.

```bash
cd train
python build_training_set.py
```

| Argument | Description |
|----------|-------------|
| `--n-per-group` | Max hate/non-hate samples per group (default: 1000) |
| `--output` | Output filename |

Output: `train/data/personahate_train_balanced_6judge.jsonl`

### Step 8: Train Models

**Train encoder models** (BERT, RoBERTa, DeBERTa, XLM-RoBERTa):

```bash
cd train
python train_encoders.py --model bert-base
```

| Model | Name |
|-------|------|
| `bert-base` | bert-base-uncased |
| `roberta-base` | roberta-base |
| `deberta-v3-base` | microsoft/deberta-v3-base |
| `xlm-roberta-base` | xlm-roberta-base |

**Train LLM with LoRA** (Llama-3.1-8B, LlamaGuard-3-8B):

```bash
cd train
python train_llm_sft.py --model llama-3.1-8b
```

| Argument | Description |
|----------|-------------|
| `--model` | Model to train |
| `--max-samples` | Limit training samples |
| `--epochs` | Number of epochs |
| `--batch-size` | Batch size |

### Step 9: Evaluate Models

**Evaluate encoder models:**

```bash
cd train
python eval_encoder.py --model-path models/bert-base_personahate/run_xxx/final --datasets all
```

| Argument | Description |
|----------|-------------|
| `--model-path` | Path to trained model |
| `--datasets` | Datasets to evaluate: `hatexplain,hatebench,davidson,mhs,newwave,all` |

**Evaluate LLM models** (requires vLLM server):

```bash
cd train
python eval_llm_sft.py --host http://localhost:8000 --model <model_name>
```

| Argument | Description |
|----------|-------------|
| `--host` | vLLM server URL |
| `--model` | Model name (auto-detect if not set) |
| `--output` | Output JSON file |
