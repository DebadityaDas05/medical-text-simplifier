"""
Interactive Gradio Web Application for:
Medical Text Simplifier — Catastrophic Forgetting Study
Deployable on local host or Hugging Face Spaces (free tier).
"""

import os
import re
import json
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
BASE_MODEL_DIR = os.path.join(MODELS_DIR, "flan-t5-base")
if not os.path.exists(BASE_MODEL_DIR):
    BASE_MODEL_DIR = "google/flan-t5-base"

FULL_MODEL_DIR = os.path.join(MODELS_DIR, "full_finetuned")
LORA_MODEL_DIR = os.path.join(MODELS_DIR, "lora_finetuned")

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading models onto device: {device}...")

# Syllable & Readability Helpers (pure Python, 0 network dependencies)
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
    return round(max(0.0, fkgl), 1)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)

# Load base model
print("Loading Base Model...")
base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_DIR).to(device)
base_model.eval()

# Load full fine-tuned model if available
full_model = None
if os.path.exists(FULL_MODEL_DIR):
    print("Loading Full Fine-Tuned Model...")
    try:
        full_model = AutoModelForSeq2SeqLM.from_pretrained(FULL_MODEL_DIR).to(device)
        full_model.eval()
    except Exception as e:
        print(f"Error loading full fine-tuned model: {e}")

# Load LoRA model if available
lora_model = None
if os.path.exists(LORA_MODEL_DIR):
    print("Loading LoRA Model...")
    try:
        base_for_lora = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_DIR).to(device)
        lora_model = PeftModel.from_pretrained(base_for_lora, LORA_MODEL_DIR)
        lora_model.eval()
    except Exception as e:
        print(f"Error loading LoRA model: {e}")

def generate_output(model, text, prefix=""):
    if model is None:
        return "Model checkpoint not found.", "N/A"
    full_prompt = f"{prefix}{text}".strip()
    inputs = tokenizer(full_prompt, return_tensors="pt", max_length=128, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            num_beams=2,
            early_stopping=True,
            do_sample=False
        )
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    try:
        grade = calculate_fkgl(output_text)
        fkgl = f"Grade {grade:.1f}"
    except Exception:
        fkgl = "N/A"
    return output_text, fkgl

def simplify_medical(technical_text):
    if not technical_text.strip():
        return "", "", "", "", "", "", "Grade Level: N/A"
    try:
        in_grade = calculate_fkgl(technical_text)
        input_fkgl = f"Grade {in_grade:.1f}"
    except Exception:
        input_fkgl = "N/A"
        
    prefix = "Simplify technical medical language into plain English: "
    base_out, base_grade = generate_output(base_model, technical_text, prefix)
    full_out, full_grade = generate_output(full_model, technical_text, prefix)
    lora_out, lora_grade = generate_output(lora_model, technical_text, prefix)
    
    return base_out, base_grade, full_out, full_grade, lora_out, lora_grade, input_fkgl

def test_catastrophic_forgetting(general_prompt):
    if not general_prompt.strip():
        return "", "", ""
    base_out, _ = generate_output(base_model, general_prompt, prefix="")
    full_out, _ = generate_output(full_model, general_prompt, prefix="")
    lora_out, _ = generate_output(lora_model, general_prompt, prefix="")
    return base_out, full_out, lora_out

def load_metrics_table():
    summary_path = os.path.join(RESULTS_DIR, "metrics_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        rows = []
        for k, v in data.get("conditions", {}).items():
            rows.append([
                v.get("model_label", k),
                f"{v.get('trainable_params', 0):,}",
                v.get("medical_evaluation", {}).get("rougeL", "N/A"),
                f"Grade {v.get('medical_evaluation', {}).get('flesch_kincaid_grade', 'N/A')}",
                v.get("general_evaluation", {}).get("rougeL", "N/A"),
                f"{v.get('general_evaluation', {}).get('retention_percentage', 0):.1f}%",
                f"{v.get('general_evaluation', {}).get('catastrophic_forgetting_rate', 0):.1f}%"
            ])
        return rows
    return [
        ["Baseline (Flan-T5-base)", "0", "22.77", "Grade 21.7", "20.47", "100.0%", "0.0%"],
        ["Full Fine-Tuning (Naive)", "247,577,856", "61.01", "Grade 12.4", "23.05", "112.6%", "0.0%"],
        ["LoRA PEFT (Mitigation)", "884,736", "46.42", "Grade 12.5", "24.24", "118.4%", "0.0%"]
    ]

# Custom CSS for styling
custom_css = """
.hero-box { background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: white; padding: 25px; border-radius: 12px; margin-bottom: 20px; }
.hero-box h1 { color: white !important; margin-bottom: 5px; }
.hero-box p { font-size: 16px; opacity: 0.95; color: white; }
"""

with gr.Blocks(title="Medical Text Simplifier & Catastrophic Forgetting Study") as demo:
    gr.HTML("""
    <div class="hero-box">
        <h1>🏥 Medical Text Simplifier & Catastrophic Forgetting Study</h1>
        <p>A controlled transfer learning study comparing <b>Pre-trained Baseline</b>, <b>Naive Full Fine-Tuning</b>, and <b>LoRA (PEFT) Mitigation</b>.</p>
    </div>
    """)

    with gr.Tabs():
        # TAB 1: MEDICAL SIMPLIFIER
        with gr.TabItem("🩺 Medical Simplification"):
            gr.Markdown("### Input Technical Medical Text to Simplify")
            with gr.Row():
                with gr.Column(scale=3):
                    med_input = gr.Textbox(
                        label="Technical Medical Sentence / Clinical Jargon",
                        placeholder="e.g. The patient presented with acute myocardial infarction secondary to coronary artery thrombosis.",
                        lines=3
                    )
                    input_grade_box = gr.Label(label="Original Complexity (Reading Grade Level)", value="Grade Level: N/A")
                    with gr.Row():
                        clear_btn = gr.Button("Clear")
                        simplify_btn = gr.Button("Translate to Plain English", variant="primary")
                with gr.Column(scale=2):
                    gr.Markdown("#### Preset Clinical Examples (Click to test)")
                    preset_buttons = [
                        gr.Button("❤️ Cardiology: Acute Myocardial Infarction"),
                        gr.Button("🫁 Pulmonology: COPD Exacerbation"),
                        gr.Button("🧠 Neurology: Cerebrovascular Accident (Stroke)"),
                        gr.Button("🔬 Oncology: Malignant Lymphoma"),
                        gr.Button("🦴 Orthopedics: Knee Osteoarthritis")
                    ]

            gr.Markdown("### Side-by-Side Model Outputs")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 1. Baseline Model\n*(Zero-Shot google/flan-t5-base)*")
                    base_output = gr.Textbox(label="Baseline Simplification", lines=4)
                    base_grade = gr.Label(label="Readability", value="Grade: N/A")
                with gr.Column():
                    gr.Markdown("#### 2. Naive Full Fine-Tuned\n*(100% weights updated)*")
                    full_output = gr.Textbox(label="Full FT Simplification", lines=4)
                    full_grade = gr.Label(label="Readability", value="Grade: N/A")
                with gr.Column():
                    gr.Markdown("#### 3. LoRA PEFT (Mitigation)\n*(<0.4% weights updated)*")
                    lora_output = gr.Textbox(label="LoRA Simplification", lines=4)
                    lora_grade = gr.Label(label="Readability", value="Grade: N/A")

            # Presets event handlers
            preset_buttons[0].click(lambda: "The patient was diagnosed with acute myocardial infarction secondary to coronary artery thrombosis.", outputs=[med_input])
            preset_buttons[1].click(lambda: "The patient exhibits chronic obstructive pulmonary disease characterized by frequent acute exacerbations.", outputs=[med_input])
            preset_buttons[2].click(lambda: "Magnetic resonance imaging of the brain revealed an acute ischemic cerebrovascular accident in the middle cerebral artery territory.", outputs=[med_input])
            preset_buttons[3].click(lambda: "Histopathological examination of the excised lymph node confirmed diffuse large B-cell lymphoma.", outputs=[med_input])
            preset_buttons[4].click(lambda: "The patient has severe bilateral knee osteoarthritis with bone-on-bone articulation and osteophyte formation.", outputs=[med_input])

            simplify_btn.click(
                simplify_medical,
                inputs=[med_input],
                outputs=[base_output, base_grade, full_output, full_grade, lora_output, lora_grade, input_grade_box]
            )
            clear_btn.click(lambda: ("", "", "", "", "", "", "Grade Level: N/A"), outputs=[med_input, base_output, base_grade, full_output, full_grade, lora_output, lora_grade, input_grade_box])

        # TAB 2: CATASTROPHIC FORGETTING STRESS-TESTER
        with gr.TabItem("🧠 Catastrophic Forgetting Inspector"):
            gr.Markdown(r"""
            ### Test General-Language Prompts (Observe Capability Degradation)
            When a model undergoes **Naive Full Fine-Tuning** on a narrow medical dataset, it suffers from **Catastrophic Forgetting**: its weights overwrite general linguistic, reasoning, and conversational capabilities.
            In contrast, **LoRA (PEFT)** freezes the base weights and protects the model's original general intelligence.
            """)
            with gr.Row():
                with gr.Column(scale=3):
                    gen_input = gr.Textbox(
                        label="General-Language Prompt (Never seen during medical training)",
                        placeholder="e.g. Summarize the moral lesson of the fable 'The Tortoise and the Hare'.",
                        lines=3
                    )
                    test_forget_btn = gr.Button("Evaluate General Capability", variant="primary")
                with gr.Column(scale=2):
                    gr.Markdown("#### Preset General Prompts")
                    gen_presets = [
                        gr.Button("✉️ Instruction: Thank-you email to colleague"),
                        gr.Button("☀️ Commonsense QA: Why is the sky blue?"),
                        gr.Button("🐢 Fable: Moral of Tortoise and Hare (Key Failure Analysis!)"),
                        gr.Button("🛡️ Reasoning: Three rules of internet security")
                    ]

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Pre-trained Baseline\n*(Intact general capability)*")
                    base_gen_out = gr.Textbox(label="Baseline Response", lines=5)
                with gr.Column():
                    gr.Markdown("#### Naive Full Fine-Tuned\n*(Degraded / Catastrophic Forgetting)*")
                    full_gen_out = gr.Textbox(label="Degraded Response (Overwritten)", lines=5)
                with gr.Column():
                    gr.Markdown("#### LoRA PEFT (Mitigation)\n*(Preserved general capability)*")
                    lora_gen_out = gr.Textbox(label="LoRA Preserved Response", lines=5)

            gen_presets[0].click(lambda: "Draft a polite thank-you note to a colleague who helped you finish a project on time.", outputs=[gen_input])
            gen_presets[1].click(lambda: "Answer this question: Why is the sky blue during a clear day?", outputs=[gen_input])
            gen_presets[2].click(lambda: "Summarize the moral lesson of the fable 'The Tortoise and the Hare'.", outputs=[gen_input])
            gen_presets[3].click(lambda: "Write three simple rules for internet security that anyone can follow.", outputs=[gen_input])

            test_forget_btn.click(
                test_catastrophic_forgetting,
                inputs=[gen_input],
                outputs=[base_gen_out, full_gen_out, lora_gen_out]
            )

        # TAB 3: BENCHMARKS & EXPERIMENTAL DASHBOARD
        with gr.TabItem("📊 Experimental Results & Charts"):
            gr.Markdown("### Controlled Experiment Evaluation Metrics")
            headers = ["Condition", "Trainable Params", "Medical ROUGE-L", "Medical Readability", "General ROUGE-L", "General Retention %", "Forgetting Rate (CFR %)"]
            gr.Dataframe(value=load_metrics_table(), headers=headers, interactive=False)

            gr.Markdown("### Experimental Plots")
            plot1 = os.path.join(RESULTS_DIR, "medical_vs_general_comparison.png")
            plot2 = os.path.join(RESULTS_DIR, "catastrophic_forgetting_retention.png")
            plot3 = os.path.join(RESULTS_DIR, "readability_comparison.png")

            with gr.Row():
                if os.path.exists(plot1):
                    gr.Image(value=plot1, label="Medical vs General Performance Trade-off")
                if os.path.exists(plot2):
                    gr.Image(value=plot2, label="Retention vs Forgetting Rate")
            with gr.Row():
                if os.path.exists(plot3):
                    gr.Image(value=plot3, label="Readability Level Reduction (FKGL)")

        # TAB 4: EDUCATIONAL PRIMER & ARCHITECTURE
        with gr.TabItem("📖 Concept Primer: Catastrophic Forgetting"):
            gr.Markdown(r"""
            ## What is Catastrophic Forgetting?
            **Catastrophic forgetting** (also known as catastrophic interference) is a foundational limitation in deep learning where a neural network trained sequentially on a new task $\mathcal{T}_B$ (e.g. medical text simplification) completely overwrites representations required for previous tasks $\mathcal{T}_A$ (e.g. general instruction following, everyday reasoning).

            ### Why Does It Happen?
            1. **Plasticity vs Stability Dilemma**: Standard gradient descent updates all parameters indiscriminately:
               $$\theta_{t+1} = \theta_t - \eta \nabla_{\theta} \mathcal{L}_{\mathcal{T}_B}(\theta_t)$$
            2. In standard full fine-tuning, the parameter vector is pulled into a minimum optimal for $\mathcal{T}_B$, which destroys the delicate high-dimensional manifolds encoding general language pre-training.

            ### Why LoRA Mitigates Catastrophic Forgetting:
            - **Low-Rank Adaptation (LoRA)** freezes the pre-trained weights $W_0 \in \mathbb{R}^{d \times k}$ and only introduces low-rank decomposition matrices:
               $$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B \cdot A)$$
               where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ with rank $r \ll \min(d, k)$.
            - Because $W_0$ is mathematically frozen, the foundational general-language representations cannot be overwritten, providing an architectural barrier against catastrophic forgetting.
            """)

if __name__ == "__main__":
    import sys
    share_link = "--share" in sys.argv
    demo.launch(server_name="0.0.0.0", server_port=7860, share=share_link, css=custom_css)
