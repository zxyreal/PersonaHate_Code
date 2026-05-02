#!/usr/bin/env python3
"""
SFT fine-tune Llama-3.1-8B and LlamaGuard-3-8B on PersonaHate for hate speech classification.
Uses LoRA for parameter-efficient fine-tuning.
"""

import argparse
import json
import torch
import pandas as pd
from pathlib import Path
from datetime import datetime
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).parent

# Load config
CONFIG_PATH = ROOT_DIR / "config.json"
if CONFIG_PATH.exists():
    with open(CONFIG_PATH, 'r') as f:
        CONFIG = json.load(f).get("train_llm_sft", {})
else:
    CONFIG = {}

DATA_PATH = ROOT_DIR / CONFIG.get("data_path", "data/personahate_train_balanced_6judge.jsonl")
OUTPUT_DIR = ROOT_DIR / CONFIG.get("output_dir", "models")

MODELS = {
    "llama-3.1-8b": {
        "name": "meta-llama/Llama-3.1-8B-Instruct",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "max_length": 1024,
        "use_system": True,
    },
    "llamaguard-3-8b": {
        "name": "meta-llama/Llama-Guard-3-8B",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "max_length": 1024,
        "use_system": False,
    },
}

SYSTEM_PROMPT = "You are a hate speech classifier. Analyze the given text and respond with only 'yes' if it is hate speech, or 'no' if it is not."

USER_TEMPLATE = """Is the following text hate speech? Hate speech is content that expresses hatred or promotes violence against people based on race, ethnicity, religion, gender, sexual orientation, disability, or origin.

Text: "{text}"

Answer with only one word: yes or no"""


def create_sft_dataset(tokenizer, use_system=True, max_samples=None):
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_json(DATA_PATH, lines=True)
    df = df[df['text'].notna() & (df['text'].str.len() > 0)]
    df['text'] = df['text'].astype(str)

    if "split_label" in df.columns and "label" not in df.columns:
        df["label"] = df["split_label"]

    print(f"Total: {len(df):,}, Hate: {(df['label']==1).sum():,}, Non-hate: {(df['label']==0).sum():,}")

    if max_samples and len(df) > max_samples:
        hate_df = df[df['label'] == 1].sample(n=min(max_samples // 2, (df['label'] == 1).sum()), random_state=42)
        nonhate_df = df[df['label'] == 0].sample(n=min(max_samples // 2, (df['label'] == 0).sum()), random_state=42)
        df = pd.concat([hate_df, nonhate_df]).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Sampled to: {len(df):,}")

    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['label'])
    print(f"Train: {len(train_df):,}, Test: {len(test_df):,}")

    def create_conversation(row):
        user_content = USER_TEMPLATE.format(text=row['text'][:1500])
        assistant_content = "yes" if row['label'] == 1 else "no"

        if use_system:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ]
        else:
            messages = [
                {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{user_content}"},
                {"role": "assistant", "content": assistant_content}
            ]

        formatted = tokenizer.apply_chat_template(messages, tokenize=False)
        return {"text": formatted}

    train_data = [create_conversation(row) for _, row in train_df.iterrows()]
    test_data = [create_conversation(row) for _, row in test_df.iterrows()]

    return Dataset.from_list(train_data), Dataset.from_list(test_data)


def train_model(model_key, max_samples=None, epochs=2, batch_size=4, grad_accum=8):
    if model_key not in MODELS:
        print(f"Unknown model: {model_key}. Available: {list(MODELS.keys())}")
        return

    config = MODELS[model_key]
    model_name = config["name"]

    print("=" * 70)
    print(f"Training: {model_key} ({model_name})")
    print(f"Dataset: {DATA_PATH.name}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_system = config.get("use_system", True)
    train_dataset, test_dataset = create_sft_dataset(
        tokenizer, use_system=use_system, max_samples=max_samples
    )

    print(f"\nLoading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=config["target_modules"]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    output_dir = OUTPUT_DIR / f"{model_key}_personahate_6judge" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        bf16=True,
        max_length=config["max_length"],
        packing=False,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
    )

    print("\nStarting SFT training...")
    trainer.train()

    # Save LoRA adapter
    lora_dir = output_dir / "lora"
    trainer.save_model(str(lora_dir))
    tokenizer.save_pretrained(str(lora_dir))

    # Merge LoRA into base model
    print("\nMerging LoRA weights...")
    merged_model = model.merge_and_unload()
    merged_dir = output_dir / "merged"
    merged_model.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))

    # Quick test
    print("\n" + "=" * 70)
    print("Quick Test")
    print("=" * 70)

    test_texts = [
        "Build the wall, and put these spics in prison with the nogs.",
        "I love spending time with my family on weekends.",
        "All Muslims are terrorists and should be deported.",
    ]

    model.eval()
    for text in test_texts:
        user_content = USER_TEMPLATE.format(text=text)
        if use_system:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
        else:
            messages = [
                {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{user_content}"},
            ]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)

        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        print(f"Text: {text[:60]}...")
        print(f"Response: {response}\n")

    print(f"\nDone! LoRA: {lora_dir}")
    print(f"Merged: {merged_dir}")
    return str(merged_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=list(MODELS.keys()))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    args = parser.parse_args()

    train_model(args.model, args.max_samples, args.epochs, args.batch_size, args.grad_accum)


if __name__ == "__main__":
    main()
