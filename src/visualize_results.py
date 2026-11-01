"""
Visualization & Reporting Script for Medical Text Simplifier & Catastrophic Forgetting Study
Generates:
1. results/medical_vs_general_comparison.png (Grouped bar chart of ROUGE-L)
2. results/catastrophic_forgetting_retention.png (Retention % & Forgetting Rate)
3. results/readability_comparison.png (FKGL Grade Level reduction)
4. results/qualitative_comparison.md (Side-by-side output case studies)
"""

import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_DIR = os.path.join(BASE_DIR, "data")

def main():
    summary_path = os.path.join(RESULTS_DIR, "metrics_summary.json")
    if not os.path.exists(summary_path):
        print(f"Error: {summary_path} does not exist yet. Run evaluate_all.py first.")
        return

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    conditions = summary["conditions"]
    labels = [v["model_label"] for v in conditions.values()]
    med_rougeL = [v["medical_evaluation"]["rougeL"] for v in conditions.values()]
    gen_rougeL = [v["general_evaluation"]["rougeL"] for v in conditions.values()]
    retention = [v["general_evaluation"]["retention_percentage"] for v in conditions.values()]
    cfr = [v["general_evaluation"]["catastrophic_forgetting_rate"] for v in conditions.values()]
    med_fkgl = [v["medical_evaluation"].get("flesch_kincaid_grade", 0.0) for v in conditions.values()]

    # Set aesthetic style
    sns.set_theme(style="whitegrid", font="sans-serif")

    # -------------------------------------------------------------
    # Plot 1: Medical vs General ROUGE-L Comparison
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    x = range(len(labels))
    width = 0.35

    bars1 = plt.bar([p - width/2 for p in x], med_rougeL, width, label='Medical Simplification (ROUGE-L)', color='#2b5c8f')
    bars2 = plt.bar([p + width/2 for p in x], gen_rougeL, width, label='General-Language Benchmark (ROUGE-L)', color='#d95f02')

    plt.ylabel('ROUGE-L Score (F1 %)', fontsize=12, fontweight='bold')
    plt.title('Medical Simplification vs General Language Performance\n(Catastrophic Forgetting Trade-Off)', fontsize=14, fontweight='bold', pad=15)
    plt.xticks(x, labels, fontsize=10, fontweight='bold')
    plt.ylim(0, 100)
    plt.legend(fontsize=11)

    # Add data labels
    for bar in bars1:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f'{yval:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#2b5c8f')
    for bar in bars2:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f'{yval:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#d95f02')

    plt.tight_layout()
    plot1_path = os.path.join(RESULTS_DIR, "medical_vs_general_comparison.png")
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"Saved: {plot1_path}")

    # -------------------------------------------------------------
    # Plot 2: Retention % and Catastrophic Forgetting Rate
    # -------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    x = range(len(labels))
    color_ret = '#2ca02c'
    color_cfr = '#d62728'

    bars_ret = ax1.bar([p - 0.2 for p in x], retention, 0.38, label='General Intelligence Retention (%)', color=color_ret, alpha=0.9)
    bars_cfr = ax1.bar([p + 0.2 for p in x], cfr, 0.38, label='Catastrophic Forgetting Rate (CFR %)', color=color_cfr, alpha=0.9)

    ax1.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax1.set_title('General Capability Retention vs Catastrophic Forgetting Rate', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10, fontweight='bold')
    ax1.set_ylim(0, 115)
    ax1.legend(fontsize=11)

    for bar in bars_ret:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f'{yval:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1b611b')
    for bar in bars_cfr:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f'{yval:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#991b1b')

    plt.tight_layout()
    plot2_path = os.path.join(RESULTS_DIR, "catastrophic_forgetting_retention.png")
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"Saved: {plot2_path}")

    # -------------------------------------------------------------
    # Plot 3: Readability Improvement (Flesch-Kincaid Grade Level)
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    input_fkgl = summary.get("input_medical_readability", {}).get("flesch_kincaid_grade", 14.5)
    read_labels = ["Original Technical Text"] + labels
    read_values = [input_fkgl] + med_fkgl
    colors = ['#7f7f7f', '#aec7e8', '#1f77b4', '#2ca02c']

    bars3 = plt.bar(range(len(read_labels)), read_values, color=colors, width=0.55)
    plt.ylabel('Flesch-Kincaid Grade Level (FKGL)', fontsize=12, fontweight='bold')
    plt.title('Medical Text Complexity Reduction\n(Lower Grade Level = More Accessible to Patients)', fontsize=13, fontweight='bold', pad=15)
    plt.xticks(range(len(read_labels)), [l.replace(" ", "\n") for l in read_labels], fontsize=9, fontweight='bold')
    plt.axhline(y=8.0, color='r', linestyle='--', alpha=0.7, label='Target Reading Level (Grade 8)')
    plt.ylim(0, max(read_values) + 3)
    plt.legend(fontsize=10)

    for bar in bars3:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f'Grade {yval:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plot3_path = os.path.join(RESULTS_DIR, "readability_comparison.png")
    plt.savefig(plot3_path, dpi=300)
    plt.close()
    print(f"Saved: {plot3_path}")

    # -------------------------------------------------------------
    # Qualitative Comparison Markdown Report
    # -------------------------------------------------------------
    preds_path = os.path.join(RESULTS_DIR, "benchmark_predictions.json")
    if os.path.exists(preds_path):
        with open(preds_path, "r", encoding="utf-8") as f:
            preds_data = json.load(f)

        medical_test_path = os.path.join(DATA_DIR, "medical_test.jsonl")
        general_bench_path = os.path.join(DATA_DIR, "general_eval_benchmark.jsonl")

        with open(medical_test_path, "r", encoding="utf-8") as f:
            med_items = [json.loads(l) for l in f if l.strip()]
        with open(general_bench_path, "r", encoding="utf-8") as f:
            gen_items = [json.loads(l) for l in f if l.strip()]

        md_content = ["# Qualitative Study: Catastrophic Forgetting & Simplification\n\n"]
        md_content.append("## 1. Medical Simplification Task (Domain Adaptation)\n")
        md_content.append("| # | Technical Medical Input | Pre-trained Baseline | Full Fine-Tuning | LoRA PEFT (Mitigation) | Gold Target |\n")
        md_content.append("|---|---|---|---|---|---|\n")

        # Select 4 representative medical samples
        for i in range(min(4, len(med_items))):
            inp = med_items[i]["input_text"].replace("Simplify technical medical language into plain English: ", "")
            tgt = med_items[i]["target_text"]
            b_pred = preds_data.get("baseline", {}).get("medical", [""])[i]
            f_pred = preds_data.get("full_finetuned", {}).get("medical", [""])[i]
            l_pred = preds_data.get("lora_finetuned", {}).get("medical", [""])[i]
            md_content.append(f"| {i+1} | {inp} | {b_pred} | {f_pred} | {l_pred} | {tgt} |\n")

        md_content.append("\n## 2. General-Language Benchmark (Catastrophic Forgetting Analysis)\n")
        md_content.append("| # | General Evaluation Prompt | Pre-trained Baseline (Original) | Full Fine-Tuned (Degraded / Catastrophic Forgetting) | LoRA PEFT (Preserved Intelligence) | Reference |\n")
        md_content.append("|---|---|---|---|---|---|\n")

        # Select 5 representative general samples
        gen_indices = [0, 15, 36, 45, 70]  # simplification, reasoning, and instruction tasks
        for rank, idx in enumerate(gen_indices, start=1):
            if idx < len(gen_items):
                inp = gen_items[idx]["input_text"]
                tgt = gen_items[idx]["target_text"]
                b_pred = preds_data.get("baseline", {}).get("general", [""])[idx]
                f_pred = preds_data.get("full_finetuned", {}).get("general", [""])[idx]
                l_pred = preds_data.get("lora_finetuned", {}).get("general", [""])[idx]
                md_content.append(f"| {rank} | {inp} | {b_pred} | {f_pred} | {l_pred} | {tgt} |\n")

        qual_path = os.path.join(RESULTS_DIR, "qualitative_comparison.md")
        with open(qual_path, "w", encoding="utf-8") as f:
            f.writelines(md_content)
        print(f"Saved: {qual_path}")

if __name__ == "__main__":
    main()

