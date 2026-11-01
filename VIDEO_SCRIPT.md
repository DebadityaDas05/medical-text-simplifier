# Video Presentation Script: Medical Text Simplifier & Catastrophic Forgetting Study
**Target Duration:** 8–12 Minutes  
**Project:** Medical Text Simplifier — Catastrophic Forgetting Study  
**Model:** `google/flan-t5-base` (248M Parameters)  
**Mitigation Strategy:** PEFT / LoRA (Low-Rank Adaptation)  

---

## Slide & Speaking Outline (Parts 1–11)

### Part 1: Real-World Problem & Motivation (0:00 - 1:15)
- **Visual:** Slide showing a side-by-side: Doctor's discharge summary vs. Patient-friendly translation.
- **Key Points:**
  - **The Problem:** Healthcare records and clinical studies are laden with dense medical terminology (e.g., *"acute myocardial infarction secondary to coronary artery thrombosis"*).
  - **The Users:** Patients, elderly individuals, caregivers, and non-clinical staff who need actionable health information at a 6th–8th grade reading level.
  - **Input / Output:**
    - *Input:* Technical medical note / pathology report.
    - *Output:* Fluent, accessible plain-English explanation without sacrificing clinical truth.
  - **Why Fine-Tuning?** General zero-shot models often either regurgitate complex Latinate jargon or oversimplify and drop critical facts. Fine-tuning teaches the model clinical translation mappings.

---

### Part 2: Core Concept — Catastrophic Forgetting (1:15 - 2:30)
- **Visual:** Diagram of Neural Weight Space showing pre-trained manifold vs. shifted weights after fine-tuning.
- **Key Points:**
  - **Definition:** Catastrophic forgetting (catastrophic interference) happens when a neural network trained sequentially on a new specific domain ($\mathcal{D}_{med}$) completely overwrites knowledge and reasoning learned during its broad pre-training.
  - **The Stability-Plasticity Dilemma:**
    - Gradient descent updates all model weights to minimize medical loss: $\theta \leftarrow \theta - \eta \nabla \mathcal{L}_{med}$.
    - As parameters shift into the medical optimum, the multi-task circuits that solved general reasoning, instruction following, and everyday tasks are disrupted.
  - **Connection to Our Study:** We explicitly test whether fine-tuning on medical simplification damages the model's performance on general language tasks, and how parameter isolation (LoRA) prevents this.

---

### Part 3: Dataset Architecture & Curation (2:30 - 3:45)
- **Visual:** Dataset architecture diagram + JSON schema code block + sample card.
- **Key Points:**
  - **Medical Dataset ($\mathcal{D}_{med}$):** 800 high-quality technical-to-plain pairs across 10 medical specialties (Cardiology, Pulmonology, Neurology, Gastroenterology, Nephrology, Oncology, Endocrinology, Infectious Disease, Orthopedics, Ophthalmology).
  - **Splits:** 650 Train (81.25%), 75 Validation (9.375%), 75 Test (9.375%).
  - **Strictly Held-Out General Benchmark ($\mathcal{D}_{gen}$):** 100 diverse general-purpose prompts:
    - 35 Everyday sentence simplification prompts.
    - 35 Commonsense & factual reasoning questions.
    - 30 Instruction-following and conversational tasks.
  - **Integrity Guarantee:** $\mathcal{D}_{gen}$ was completely isolated and never exposed to the models during training.

---

### Part 4: Model Selection & Hardware Justification (3:45 - 4:45)
- **Visual:** Flan-T5 architecture diagram (Encoder-Decoder) & Specs table.
- **Key Points:**
  - **Base Model:** `google/flan-t5-base` (248M parameters).
  - **Why Flan-T5?**
    - Encoder-Decoder (Seq2Seq) is the premier architecture for text rewriting and translation.
    - Instruction-tuned on 1,800+ tasks, providing a proven, rich general-language benchmark baseline.
  - **Hardware:** Local NVIDIA GeForce RTX 5050 Laptop GPU (8 GB VRAM, CUDA 12.8), utilizing FP16 mixed precision for fast training without cloud subscription fees.

---

### Part 5: Code Walkthrough (4:45 - 6:00)
- **Visual:** Screen recording showing `src/train_full.py`, `src/train_lora.py`, and `src/evaluate_all.py`.
- **Key Points:**
  - **Where Fine-Tuning Actually Happens:**
    - In `train_full.py`: All 247.5M weights are set to `requires_grad=True`.
    - In `train_lora.py`: PEFT `LoraConfig(r=8, lora_alpha=16, target_modules=['q', 'v'])` freezes 99.64% of weights; only low-rank adapter matrices $A$ and $B$ (884k params, 0.35%) are trained.
  - **Data Collator:** `DataCollatorForSeq2Seq` dynamically pads sequences and masks label padding with `-100`.

---

### Part 6: Training Setup & Hyperparameters (6:00 - 7:00)
- **Visual:** Hyperparameter comparison table: Full Fine-Tuning vs. LoRA PEFT.
- **Key Points:**
  - **Full Fine-Tuning:**
    - Epochs: 4 | Batch Size: 8 | Learning Rate: $5 \times 10^{-5}$ | Optimizer: AdamW | FP16.
  - **LoRA PEFT:**
    - Epochs: 4 | Batch Size: 8 | Learning Rate: $3 \times 10^{-4}$ | Rank $r=8$, Alpha $\alpha=16$.
  - **Loss Dynamics:** Both models converge smoothly from cross-entropy loss $>2.5$ down to $<0.30$.

---

### Part 7: Controlled Experiment Design (7:00 - 7:45)
- **Visual:** Controlled experiment matrix.
- **Key Points:**
  - **Independent Variable:** Parameter update scope (Full 100% parameter update vs. 0.35% LoRA adapter update).
  - **Controlled Constants:** Base model (`google/flan-t5-base`), training dataset, number of epochs (4), batch size (8), evaluation benchmark test sets, and evaluation metrics.
  - **Three Test Conditions:** Baseline (Untrained) vs. Condition 1 (Full FT) vs. Condition 2 (LoRA Mitigation).

---

### Part 8: Quantitative Results & Visualizations (7:45 - 9:00)
- **Visual:** Display `results/medical_vs_general_comparison.png` and `results/catastrophic_forgetting_retention.png`.
- **Key Points:**
  - **Medical Domain Improvement:**
    - Baseline: ROUGE-L ~42.8%
    - Full FT: ROUGE-L ~78.4% (+35.6% gain!)
    - LoRA PEFT: ROUGE-L ~76.9% (+34.1% gain!)
  - **Catastrophic Forgetting on General Benchmark:**
    - Baseline: 100% retention (reference ROUGE-L ~58.6%).
    - Full FT: General score plunges down to ~24.1% (**Catastrophic Forgetting Rate: 58.9%**).
    - LoRA PEFT: General score maintained at ~57.2% (**General Retention: 97.6%**, Forgetting Rate: 2.4%).
  - **Readability (FKGL):** Medical text complexity drops from College level (Grade 14.5) down to Middle School (Grade 7.2).

---

### Part 9: Failure Analysis (9:00 - 9:45)
- **Visual:** Highlight a specific failure case from `results/qualitative_comparison.md`.
- **Key Points:**
  - **Failure Mode 1: Medical Hallucination / Bias in General Tasks under Full FT:**
    - When asked a general prompt like *"Draft a polite thank-you note to a colleague"*, the Naive Full Fine-Tuned model hallucinates clinical terminology or outputs disjointed medical sentence fragments (e.g., *"Thank you for your assistance with the surgical admission evaluation"*).
    - **Reason:** The unconstrained parameter drift destroyed the general language conditional probability distribution, biasing every activation toward medical syntax.
  - **Contrast:** LoRA generates an appropriate, warm, general thank-you note because its base language distribution remains intact.

---

### Part 10: Mandatory Live Demo (9:45 - 11:00)
- **Visual:** Switch to the live Gradio interface (`app.py`).
- **Actions on Camera:**
  1. **Medical Simplification:** Paste a fresh unseen medical sentence: *"Patient presents with progressive orthopnea and nocturnal paroxysmal dyspnea indicative of left heart failure."*
     - Show Baseline output (stiff, repeating jargon).
     - Show Full FT and LoRA outputs (plain English: *"The patient has trouble breathing when lying flat and wakes up breathless at night from heart failure"*).
  2. **Catastrophic Forgetting Stress-Test:** Enter an unseen general prompt: *"Answer this question: Why do airplanes have curved wings?"*
     - Show how Full FT stumbles or degenerates.
     - Show how LoRA produces a clear, accurate physics explanation.

---

### Part 11: Conclusion & Future Work (11:00 - 11:45)
- **Visual:** Summary takeaway slide.
- **Key Points:**
  - **Key Finding:** Parameter-Efficient Fine-Tuning (LoRA) is not just a computational shortcut—it is a powerful architectural defense against catastrophic forgetting.
  - **Limitations:** LoRA still requires careful selection of rank $r$; medical simplification models must be paired with clinical safety guardrails before patient deployment.
  - **Future Improvement:** Incorporating Fact-Checking (Clinical Hallucination Detection) via Entailment / Med-NLI models.

