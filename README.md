---
title: Medical Text Simplifier
emoji: 🏥
colorFrom: blue
colorTo: indigo
sdk: static
pinned: false
---

# Medical Text Simplifier — Catastrophic Forgetting Study

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face%20Spaces-Live%20Demo-yellow.svg)](https://huggingface.co/spaces/FuriousSlayer/medical-text-simplifier)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11%2BCUDA-red.svg)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/🤗%20Transformers-4.40+-yellow.svg)](https://huggingface.co/transformers)
[![PEFT / LoRA](https://img.shields.io/badge/PEFT-LoRA%20(r%3D8)-green.svg)](https://github.com/huggingface/peft)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

> 🌐 **Live Interactive Deployment (100% Free Forever):**  
> **[https://huggingface.co/spaces/FuriousSlayer/medical-text-simplifier](https://huggingface.co/spaces/FuriousSlayer/medical-text-simplifier)**

A controlled transfer learning study investigating **Catastrophic Forgetting** when specializing a general-language model (`google/flan-t5-base`) into a technical medical text simplifier, and demonstrating mitigation using **Low-Rank Adaptation (LoRA / PEFT)**.

---

## 📋 Table of Contents
1. [Project Overview & Real-World Use Case](#-project-overview--real-world-use-case)
2. [The Core Concept: Catastrophic Forgetting](#-the-core-concept-catastrophic-forgetting)
3. [Controlled Experimental Design](#-controlled-experimental-design)
4. [Dataset Construction & Curation](#-dataset-construction--curation)
5. [Model Architecture & Selection Justification](#-model-architecture--selection-justification)
6. [Quantitative Results & Findings](#-quantitative-results--findings)
7. [Qualitative Analysis & Failure Modes](#-qualitative-analysis--failure-modes)
8. [Reproducibility & Step-by-Step Instructions](#-reproducibility--step-by-step-instructions)
9. [Free Cloud Live Deployment (Hugging Face Spaces)](#-free-cloud-live-deployment-hugging-face-spaces)
10. [Video Presentation Guide (Parts 1–11)](#-video-presentation-guide-parts-111)

---

## 🏥 Project Overview & Real-World Use Case

- **The Problem:** Clinical notes, hospital discharge summaries, and medical research are written in dense, Latinate terminology (e.g., *"acute myocardial infarction secondary to coronary artery thrombosis"*). For patients, caregivers, and non-clinical readers, this creates a severe comprehension barrier.
- **The Real-World Solution:** An automated medical text simplifier that translates technical jargon into accessible, plain-English explanations (targeting a 6th–8th grade reading level) while maintaining 100% factual accuracy.
- **The Core Scientific Tension:** While fine-tuning enables the model to understand complex medical concepts, does specializing on this narrow domain cause the model to forget its foundational general-language reasoning, commonsense QA, and instruction-following abilities?

---

## 🧠 The Core Concept: Catastrophic Forgetting

### What is Catastrophic Forgetting?
**Catastrophic forgetting** (or *catastrophic interference*) occurs when an artificial neural network trained on a task $\mathcal{T}_A$ is sequentially adapted to a new task $\mathcal{T}_B$. As standard gradient descent updates all parameter weights to minimize loss on $\mathcal{T}_B$:
$$\theta_{t+1} = \theta_t - \eta \nabla_{\theta} \mathcal{L}_{\mathcal{T}_B}(\theta_t)$$
the network's weights are pulled into a subspace that **overwrites** the representations and attention patterns required for $\mathcal{T}_A$.

### The Stability-Plasticity Dilemma:
- **Plasticity:** The capacity to integrate new knowledge (medical simplification).
- **Stability:** The capacity to preserve previously acquired knowledge (general language).

### Why LoRA Mitigates Forgetting:
Instead of updating all 247.5M weights, **Low-Rank Adaptation (LoRA)** freezes the pre-trained weights $W_0 \in \mathbb{R}^{d \times k}$ and only trains rank decomposition matrices:
$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B \cdot A)$$
where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ with rank $r=8 \ll \min(d, k)$.
Because the base weights $W_0$ remain mathematically locked, the foundational general-language manifolds are shielded from destruction.

---

## 🔬 Controlled Experimental Design

We evaluated three strictly controlled conditions:

| Condition | Strategy | Trainable Parameters | Description |
| :--- | :--- | :--- | :--- |
| **0. Baseline** | Zero-Shot Base Model | **0** (0.00%) | Untrained `google/flan-t5-base` directly prompted. |
| **1. Full Fine-Tuning** | Naive Sequential FT | **247,577,856** (100.00%) | All parameters updated on medical data (causes catastrophic forgetting). |
| **2. LoRA PEFT** | Parameter-Isolated Adapter | **884,736** (0.356%) | Base weights frozen; only rank-8 adapters trained (mitigation). |

**Constants Held Rigidly Across Experiments:**
- Base Architecture: `google/flan-t5-base`
- Training Data: Identical 650 medical pairs
- Epochs: 4 | Batch Size: 8 | Optimizer: AdamW | Precision: BF16
- Evaluation Sets: Identical 75-example Medical Test set & 100-example General Language Benchmark.

---

## 📊 Dataset Construction & Curation

1. **Medical Simplification Dataset ($\mathcal{D}_{med}$, 800 pairs):**
   - Curated from open clinical benchmarks (MSD, Cochrane Plain Language Summaries, MedQuAD).
   - Covers 10 distinct medical domains: Cardiology, Pulmonology, Neurology, Gastroenterology, Nephrology, Oncology, Endocrinology, Infectious Disease, Orthopedics, Ophthalmology.
   - **Splits:** 650 Train (81.25%), 75 Validation (9.375%), 75 Test (9.375%).
   - *Example:*
     - **Input:** *"Simplify technical medical language into plain English: The patient was diagnosed with acute myocardial infarction secondary to coronary artery thrombosis."*
     - **Target:** *"The patient had a sudden heart attack caused by a blood clot blocking blood flow to the heart."*

2. **General-Language Evaluation Benchmark ($\mathcal{D}_{gen}$, 100 pairs — Held-Out):**
   - **Never seen during medical training.**
   - Spans 3 domains:
     - 35 General sentence simplifications (Wikipedia/business prose).
     - 35 Factual & commonsense QA prompts.
     - 30 Everyday conversational and instruction-following tasks.

---

## ⚙️ Model Architecture & Selection Justification

- **Selected Model:** `google/flan-t5-base` (248 Million Parameters).
- **Justification:**
  1. **Encoder-Decoder Architecture:** Superior for sequence-to-sequence conditional text generation and rewriting compared to causal decoder-only models.
  2. **Instruction-Tuning Heritage:** Pre-trained on 1,800+ tasks, establishing an unambiguous, high-fidelity general-language baseline.
  3. **Zero Financial Cost & High Efficiency:** Runs and trains locally in under 3 minutes on an 8 GB NVIDIA RTX 5050 GPU using BF16 precision.

---

## 📈 Quantitative Results & Findings

### Summary Evaluation Table

| Condition | Trainable Params | Medical ROUGE-L | Medical Readability | General ROUGE-L | General Retention | Catastrophic Forgetting Rate (CFR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Flan-T5-base)** | 0 (0%) | 42.8% | Grade 11.4 | 58.6% | **100.0%** (Ref) | **0.0%** |
| **Full Fine-Tuned (Naive)** | 247.5M (100%) | **78.4%** | **Grade 7.2** | 24.1% | **41.1%** | **58.9%** (Severe) |
| **LoRA PEFT (Mitigation)** | 884K (0.356%) | **76.9%** | **Grade 7.4** | **57.2%** | **97.6%** | **2.4%** (Preserved) |

### Key Experimental Insights:
1. **Medical Specialization:** Both Full Fine-Tuning (+35.6% ROUGE-L) and LoRA (+34.1% ROUGE-L) successfully learn clinical text simplification, lowering the reading grade level from Grade 14.5 down to Grade 7.2–7.4.
2. **Catastrophic Forgetting Phenomenon:** Under Full Fine-Tuning, general-language performance collapses by **58.9%**. The model loses the ability to write emails, answer basic science questions, or summarize everyday text.
3. **LoRA Mitigation:** LoRA retains **97.6%** of the model's original general intelligence while matching Full Fine-Tuning's medical simplification performance within 1.5% ROUGE-L.

---

## 🔍 Qualitative Analysis & Failure Modes

### Failure Case 1: Over-Specialization / Medical Bias on General Tasks
- **General Prompt:** *"Draft a polite thank-you note to a colleague who helped you finish a project on time."*
- **Baseline:** *"Thank you so much for your assistance with our project. Your help was invaluable in getting everything completed on time."* (Fluent, appropriate)
- **Full Fine-Tuned (Failure Mode):** *"Discharge note: The colleague was administered assistance for project completion on time with stable vitals."*
  - **Root Cause:** Parameter drift caused by updating 100% of weights. The unconditional prior distribution is now entirely biased toward clinical discharge structures.
- **LoRA PEFT (Mitigation):** *"Thank you very much for your great help on the project. We finished on time thanks to your support."* (Correctly preserves conversational tone)

---

## 💻 Reproducibility & Step-by-Step Instructions

### 1. Environment Setup
```powershell
# Create isolated virtual environment
python -m venv --system-site-packages .venv
.\.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 2. Dataset Preparation
```powershell
python src/dataset_prep.py
```

### 3. Run Controlled Fine-Tuning Experiments
```powershell
# Experiment 1: Naive Full Fine-Tuning (Causes catastrophic forgetting)
python src/train_full.py

# Experiment 2: LoRA Mitigation Fine-Tuning (Preserves general intelligence)
python src/train_lora.py
```

### 4. Evaluate & Visualize
```powershell
# Run evaluation across all conditions and benchmarks
python src/evaluate_all.py

# Generate comparison plots and qualitative report
python src/visualize_results.py
```

### 5. Launch Interactive Web Demo (100% Free)
```powershell
python app.py
```
Open your browser at `http://localhost:7860`.

---

## 🚀 Free Cloud Live Deployment (Hugging Face Spaces)

This project is deployed and hosted permanently on Hugging Face Spaces:

🔗 **Live Public Space:** **[https://huggingface.co/spaces/FuriousSlayer/medical-text-simplifier](https://huggingface.co/spaces/FuriousSlayer/medical-text-simplifier)**

### Features Included on the Live Space:
- **Interactive Medical Simplifier**: Test clinical cases across Cardiology, Pulmonology, Stroke, Oncology, and Orthopedics with side-by-side model outputs and readability grade levels.
- **Catastrophic Forgetting Inspector**: Compare outputs on held-out general-language prompts and observe semantic degradation in naive full fine-tuning vs. preservation in LoRA PEFT.
- **Experimental Benchmarks & High-Res Plots**: Explore the quantitative evaluation metrics table and comparative visualization plots.
- **Theoretical & Mathematical Primer**: In-depth explanation of the stability-plasticity dilemma and LoRA parameter isolation.
- **Reproducibility Walkthrough**: Instructions and direct access to dataset files and trained LoRA weights.

To re-deploy or update the Space at any time:
```powershell
python deploy_hf_space.py --token <YOUR_HF_TOKEN>
```

---

## 🎥 Video Presentation Guide (Parts 1–11)

For your 8–12 minute video presentation, refer to the complete word-for-word and slide-by-slide script in [`VIDEO_SCRIPT.md`](VIDEO_SCRIPT.md).

