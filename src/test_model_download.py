import os
import sys

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from huggingface_hub import snapshot_download

MODEL_NAME = "google/flan-t5-base"
print(f"Starting snapshot download for {MODEL_NAME} with progress bar...")

path = snapshot_download(
    repo_id=MODEL_NAME,
    repo_type="model",
    ignore_patterns=["*.msgpack", "*.h5", "*.tflite", "*.ot", "flax_model.msgpack", "tf_model.h5"]
)
print(f"\nSuccessfully downloaded {MODEL_NAME} to: {path}")

# Verify AutoModel loading
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForSeq2SeqLM.from_pretrained(path)
print(f"Verified model loaded! Parameters: {model.num_parameters():,}")
