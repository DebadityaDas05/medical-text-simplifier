"""
Full Fine-Tuning Script for Medical Text Simplification (Condition 1)
Trains all parameters of google/flan-t5-base on medical_train.jsonl.
This demonstrates the standard (naive) sequential transfer learning setup,
which induces Catastrophic Forgetting on general-language tasks.
"""

import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "models", "full_finetuned")
LOCAL_BASE_DIR = os.path.join(BASE_DIR, "models", "flan-t5-base")
MODEL_NAME = LOCAL_BASE_DIR if os.path.exists(LOCAL_BASE_DIR) else "google/flan-t5-base"

def load_jsonl(filepath):
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def preprocess_function(examples, tokenizer, max_input_length=128, max_target_length=128):
    inputs = [ex for ex in examples["input_text"]]
    targets = [ex for ex in examples["target_text"]]
    
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding=False)
    labels = tokenizer(text_target=targets, max_length=max_target_length, truncation=True, padding=False)
    
    # Replace padding token id's with -100 so loss is not computed on padding
    labels["input_ids"] = [
        [(label if label != tokenizer.pad_token_id else -100) for label in label_seq]
        for label_seq in labels["input_ids"]
    ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    print("=" * 70)
    print("EXPERIMENT 1: NAIVE FULL FINE-TUNING (CATISTROPHIC FORGETTING BASELINE)")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    
    total_params = model.num_parameters()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters (100%): {trainable_params:,} ({trainable_params / total_params * 100:.2f}%)")

    # Load datasets
    train_data = load_jsonl(os.path.join(DATA_DIR, "medical_train.jsonl"))
    val_data = load_jsonl(os.path.join(DATA_DIR, "medical_val.jsonl"))
    print(f"Loaded {len(train_data)} training pairs and {len(val_data)} validation pairs.")

    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    tokenized_train = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    tokenized_val = val_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=val_dataset.column_names,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest")

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        save_total_limit=1,
        num_train_epochs=4,
        predict_with_generate=True,
        fp16=False,
        bf16=torch.cuda.is_bf16_supported(),
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        logging_steps=20,
        load_best_model_at_end=True,
        report_to="none",
        warmup_ratio=0.1,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("\nStarting Full Fine-Tuning on Medical Simplification...")
    train_result = trainer.train()
    print("Training finished!")
    print(f"Train Loss: {train_result.training_loss:.4f}")

    print(f"Saving fine-tuned model and tokenizer to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Save training summary
    summary_path = os.path.join(OUTPUT_DIR, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": MODEL_NAME,
            "training_type": "full_finetuning",
            "total_params": total_params,
            "trainable_params": trainable_params,
            "train_loss": train_result.training_loss,
            "epochs": 4,
            "batch_size": 8,
            "learning_rate": 5e-5
        }, f, indent=2)
    print(f"Saved summary to {summary_path}")

if __name__ == "__main__":
    main()
