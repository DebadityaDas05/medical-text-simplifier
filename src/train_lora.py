"""
LoRA Fine-Tuning Script for Medical Text Simplification (Condition 2: Mitigation)
Applies Parameter-Efficient Fine-Tuning (PEFT) using Low-Rank Adaptation (LoRA)
to google/flan-t5-base.
By freezing the base model's 247M+ parameters and training only low-rank adapters (~880K params, <0.4%),
the model retains its general-language capabilities while effectively specializing on medical simplification,
thereby mitigating Catastrophic Forgetting.
"""

import os
import json
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "models", "lora_finetuned")
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
    
    labels["input_ids"] = [
        [(label if label != tokenizer.pad_token_id else -100) for label in label_seq]
        for label_seq in labels["input_ids"]
    ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    print("=" * 70)
    print("EXPERIMENT 2: MITIGATION VIA PEFT / LoRA (PRESERVING GENERAL INTELLIGENCE)")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # Configure LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    trainable_params, total_params = model.get_nb_trainable_parameters()
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,} ({trainable_params / total_params * 100:.3f}%)")
    print(f"Frozen Parameters: {total_params - trainable_params:,} ({(total_params - trainable_params) / total_params * 100:.3f}%)")

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
        learning_rate=3e-4,
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

    print("\nStarting LoRA Fine-Tuning on Medical Simplification...")
    train_result = trainer.train()
    print("LoRA Training finished!")
    print(f"Train Loss: {train_result.training_loss:.4f}")

    print(f"Saving LoRA adapter and tokenizer to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Save summary
    summary_path = os.path.join(OUTPUT_DIR, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": MODEL_NAME,
            "training_type": "lora_peft",
            "total_params": total_params,
            "trainable_params": trainable_params,
            "lora_r": 8,
            "lora_alpha": 16,
            "train_loss": train_result.training_loss,
            "epochs": 4,
            "batch_size": 8,
            "learning_rate": 3e-4
        }, f, indent=2)
    print(f"Saved summary to {summary_path}")

if __name__ == "__main__":
    main()
