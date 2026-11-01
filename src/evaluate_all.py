"""
Evaluation Script for Medical Text Simplifier & Catastrophic Forgetting Study
Evaluates:
1. Baseline Model: models/flan-t5-base (Zero-shot / Untrained)
2. Full Fine-Tuned Model: models/full_finetuned (Naive transfer - Catastrophic Forgetting)
3. LoRA Fine-Tuned Model: models/lora_finetuned (Mitigation - PEFT parameter isolation)

Benchmarks:
1. Medical Test Set (data/medical_test.jsonl - 75 examples)
2. General-Language Benchmark (data/general_eval_benchmark.jsonl - 100 held-out examples)
"""

import os
import re
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from rouge_score import rouge_scorer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

LOCAL_BASE_DIR = os.path.join(MODELS_DIR, "flan-t5-base")
MODEL_NAME = LOCAL_BASE_DIR if os.path.exists(LOCAL_BASE_DIR) else "google/flan-t5-base"

def load_jsonl(filepath):
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def count_syllables_word(word):
    word = word.lower().strip()
    if not word:
        return 1
    word = re.sub(r'[^a-z]', '', word)
    if len(word) <= 3:
        return 1
    if word.endswith('e') and not word.endswith('le'):
        word = word[:-1]
    syllables = len(re.findall(r'[aeiouy]+', word))
    return max(1, syllables)

def calculate_fkgl(text):
    sentences = max(1, len([s for s in re.split(r'[.!?]+', text.strip()) if s.strip()]))
    words = [w for w in text.split() if any(c.isalnum() for c in w)]
    if not words:
        return 0.0
    total_words = len(words)
    total_syllables = sum(count_syllables_word(w) for w in words)
    fkgl = 0.39 * (total_words / sentences) + 11.8 * (total_syllables / total_words) - 15.59
    return round(max(0.0, fkgl), 2)

def calculate_fre(text):
    sentences = max(1, len([s for s in re.split(r'[.!?]+', text.strip()) if s.strip()]))
    words = [w for w in text.split() if any(c.isalnum() for c in w)]
    if not words:
        return 0.0
    total_words = len(words)
    total_syllables = sum(count_syllables_word(w) for w in words)
    fre = 206.835 - 1.015 * (total_words / sentences) - 84.6 * (total_syllables / total_words)
    return round(max(0.0, min(100.0, fre)), 2)

def compute_readability_metrics(texts):
    fkgl_scores = [calculate_fkgl(t) for t in texts if len(t.strip().split()) >= 3]
    fre_scores = [calculate_fre(t) for t in texts if len(t.strip().split()) >= 3]
    avg_fkgl = round(sum(fkgl_scores) / len(fkgl_scores), 2) if fkgl_scores else 0.0
    avg_fre = round(sum(fre_scores) / len(fre_scores), 2) if fre_scores else 0.0
    return {
        "flesch_kincaid_grade": avg_fkgl,
        "flesch_reading_ease": avg_fre
    }

def compute_rouge_scores(predictions, references):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    r1, r2, rl = [], [], []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        r1.append(scores['rouge1'].fmeasure * 100)
        r2.append(scores['rouge2'].fmeasure * 100)
        rl.append(scores['rougeL'].fmeasure * 100)
    return {
        "rouge1": round(sum(r1) / len(r1), 2),
        "rouge2": round(sum(r2) / len(r2), 2),
        "rougeL": round(sum(rl) / len(rl), 2)
    }

def generate_predictions_batch(model, tokenizer, dataset, device, batch_size=16, max_new_tokens=100):
    model.eval()
    prompts = [item["input_text"] for item in dataset]
    predictions = []
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            max_length=128,
            padding=True,
            truncation=True
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=2,
                early_stopping=True,
                do_sample=False
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend([d.strip() for d in decoded])
        
    return predictions

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} | Model path: {MODEL_NAME}")

    medical_test = load_jsonl(os.path.join(DATA_DIR, "medical_test.jsonl"))
    general_benchmark = load_jsonl(os.path.join(DATA_DIR, "general_eval_benchmark.jsonl"))
    print(f"Loaded Medical Test ({len(medical_test)}) and General Benchmark ({len(general_benchmark)}).")

    med_refs = [item["target_text"] for item in medical_test]
    gen_refs = [item["target_text"] for item in general_benchmark]

    med_raw_inputs = [item["input_text"].replace("Simplify technical medical language into plain English: ", "") for item in medical_test]
    input_readability = compute_readability_metrics(med_raw_inputs)
    print(f"Original Medical Input Readability - FKGL: Grade {input_readability['flesch_kincaid_grade']}, FRE: {input_readability['flesch_reading_ease']}")

    results_summary = {
        "input_medical_readability": input_readability,
        "conditions": {}
    }
    all_predictions = {}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 1. BASELINE
    print("\n" + "=" * 60)
    print("EVALUATING CONDITION 0: BASELINE (google/flan-t5-base)")
    print("=" * 60)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

    print("Evaluating Baseline on Medical Test...")
    base_med_preds = generate_predictions_batch(base_model, tokenizer, medical_test, device)
    base_med_rouge = compute_rouge_scores(base_med_preds, med_refs)
    base_med_read = compute_readability_metrics(base_med_preds)

    print("Evaluating Baseline on General Benchmark...")
    base_gen_preds = generate_predictions_batch(base_model, tokenizer, general_benchmark, device)
    base_gen_rouge = compute_rouge_scores(base_gen_preds, gen_refs)

    results_summary["conditions"]["baseline"] = {
        "model_label": "Pre-trained Base Model (Flan-T5-base)",
        "trainable_params": 0,
        "medical_evaluation": {
            **base_med_rouge,
            **base_med_read
        },
        "general_evaluation": {
            **base_gen_rouge,
            "retention_percentage": 100.0,
            "catastrophic_forgetting_rate": 0.0
        }
    }
    all_predictions["baseline"] = {
        "medical": base_med_preds,
        "general": base_gen_preds
    }
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. FULL FINE-TUNED
    full_path = os.path.join(MODELS_DIR, "full_finetuned")
    if os.path.exists(full_path):
        print("\n" + "=" * 60)
        print("EVALUATING CONDITION 1: FULL FINE-TUNING (Naive Transfer)")
        print("=" * 60)
        full_model = AutoModelForSeq2SeqLM.from_pretrained(full_path).to(device)
        full_tokenizer = AutoTokenizer.from_pretrained(full_path)

        print("Evaluating Full FT on Medical Test...")
        full_med_preds = generate_predictions_batch(full_model, full_tokenizer, medical_test, device)
        full_med_rouge = compute_rouge_scores(full_med_preds, med_refs)
        full_med_read = compute_readability_metrics(full_med_preds)

        print("Evaluating Full FT on General Benchmark...")
        full_gen_preds = generate_predictions_batch(full_model, full_tokenizer, general_benchmark, device)
        full_gen_rouge = compute_rouge_scores(full_gen_preds, gen_refs)

        base_r_l = results_summary["conditions"]["baseline"]["general_evaluation"]["rougeL"]
        full_retention = round((full_gen_rouge["rougeL"] / base_r_l) * 100, 2) if base_r_l > 0 else 0.0
        full_cfr = round(max(0.0, ((base_r_l - full_gen_rouge["rougeL"]) / base_r_l) * 100), 2) if base_r_l > 0 else 0.0

        results_summary["conditions"]["full_finetuned"] = {
            "model_label": "Full Fine-Tuning (Naive)",
            "trainable_params": 247577856,
            "medical_evaluation": {
                **full_med_rouge,
                **full_med_read
            },
            "general_evaluation": {
                **full_gen_rouge,
                "retention_percentage": full_retention,
                "catastrophic_forgetting_rate": full_cfr
            }
        }
        all_predictions["full_finetuned"] = {
            "medical": full_med_preds,
            "general": full_gen_preds
        }
        del full_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 3. LoRA PEFT
    lora_path = os.path.join(MODELS_DIR, "lora_finetuned")
    if os.path.exists(lora_path):
        print("\n" + "=" * 60)
        print("EVALUATING CONDITION 2: LoRA PEFT (Mitigation)")
        print("=" * 60)
        base_for_lora = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        lora_model = PeftModel.from_pretrained(base_for_lora, lora_path).to(device)
        lora_tokenizer = AutoTokenizer.from_pretrained(lora_path)

        print("Evaluating LoRA on Medical Test...")
        lora_med_preds = generate_predictions_batch(lora_model, lora_tokenizer, medical_test, device)
        lora_med_rouge = compute_rouge_scores(lora_med_preds, med_refs)
        lora_med_read = compute_readability_metrics(lora_med_preds)

        print("Evaluating LoRA on General Benchmark...")
        lora_gen_preds = generate_predictions_batch(lora_model, lora_tokenizer, general_benchmark, device)
        lora_gen_rouge = compute_rouge_scores(lora_gen_preds, gen_refs)

        base_r_l = results_summary["conditions"]["baseline"]["general_evaluation"]["rougeL"]
        lora_retention = round((lora_gen_rouge["rougeL"] / base_r_l) * 100, 2) if base_r_l > 0 else 0.0
        lora_cfr = round(max(0.0, ((base_r_l - lora_gen_rouge["rougeL"]) / base_r_l) * 100), 2) if base_r_l > 0 else 0.0

        results_summary["conditions"]["lora_finetuned"] = {
            "model_label": "LoRA PEFT (Mitigation)",
            "trainable_params": 884736,
            "medical_evaluation": {
                **lora_med_rouge,
                **lora_med_read
            },
            "general_evaluation": {
                **lora_gen_rouge,
                "retention_percentage": lora_retention,
                "catastrophic_forgetting_rate": lora_cfr
            }
        }
        all_predictions["lora_finetuned"] = {
            "medical": lora_med_preds,
            "general": lora_gen_preds
        }
        del lora_model
        del base_for_lora
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save summary json
    summary_file = os.path.join(RESULTS_DIR, "metrics_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nSaved metrics summary to {summary_file}")

    # Save predictions json
    preds_file = os.path.join(RESULTS_DIR, "benchmark_predictions.json")
    with open(preds_file, "w", encoding="utf-8") as f:
        json.dump(all_predictions, f, indent=2)
    print(f"Saved benchmark predictions to {preds_file}")

    print("\n" + "=" * 90)
    print("FINAL CONTROLLED EXPERIMENT BENCHMARK RESULTS:")
    print("=" * 90)
    print(f"{'Condition':<25} | {'Med ROUGE-L':<12} | {'Med FKGL':<10} | {'Gen ROUGE-L':<12} | {'Retention %':<12} | {'CFR %':<8}")
    print("-" * 90)
    for k, v in results_summary["conditions"].items():
        med_rl = v["medical_evaluation"]["rougeL"]
        med_fkgl = v["medical_evaluation"].get("flesch_kincaid_grade", "N/A")
        gen_rl = v["general_evaluation"]["rougeL"]
        ret = v["general_evaluation"]["retention_percentage"]
        cfr = v["general_evaluation"]["catastrophic_forgetting_rate"]
        print(f"{v['model_label']:<25} | {med_rl:<12} | Grade {med_fkgl:<4} | {gen_rl:<12} | {ret:<11}% | {cfr:<7}%")
    print("=" * 90)

if __name__ == "__main__":
    main()
