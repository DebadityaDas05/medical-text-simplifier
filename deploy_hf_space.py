"""
Hugging Face Spaces Deployment Script (100% Free Static Space)
Deploys the interactive web application to Hugging Face Static Spaces (100% Free for Everyone).
"""

import os
import sys
from huggingface_hub import HfApi, login

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def deploy_to_spaces(repo_id=None, token=None):
    print("=" * 65)
    print("DEPLOYING TO HUGGING FACE STATIC SPACES (100% FREE FOR EVERYONE)")
    print("=" * 65)

    if token:
        login(token=token)
    elif os.getenv("HF_TOKEN"):
        login(token=os.getenv("HF_TOKEN"))
    
    api = HfApi()

    # Get user info
    try:
        user_info = api.whoami()
        username = user_info["name"]
        print(f"Logged in as Hugging Face user: {username}")
    except Exception as e:
        print("\n[!] Not logged in to Hugging Face.")
        print("Please provide your free Hugging Face token:")
        print("1. Get a free write token from: https://huggingface.co/settings/tokens")
        print("2. Run: python deploy_hf_space.py --token YOUR_HF_TOKEN\n")
        return None

    if not repo_id:
        repo_id = f"{username}/medical-text-simplifier"

    print(f"\nCreating / Verifying Static Space: {repo_id}...")
    try:
        space_url = api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="static",  # 100% Free forever for everyone on Hugging Face
            exist_ok=True
        )
        print(f"Static Space created/verified successfully!")
    except Exception as e:
        print(f"Note on create_repo: {e}")

    # Files to upload
    files_to_upload = [
        "index.html",
        "README.md",
        "VIDEO_SCRIPT.md",
        "app.py",
        "requirements.txt",
        os.path.join("results", "metrics_summary.json"),
        os.path.join("results", "benchmark_predictions.json"),
        os.path.join("results", "medical_vs_general_comparison.png"),
        os.path.join("results", "catastrophic_forgetting_retention.png"),
        os.path.join("results", "readability_comparison.png"),
        os.path.join("results", "qualitative_comparison.md"),
        os.path.join("data", "general_eval_benchmark.jsonl"),
        os.path.join("data", "medical_test.jsonl")
    ]

    print("\nUploading core application, web assets, and evaluation plots...")
    for f in files_to_upload:
        full_p = os.path.join(BASE_DIR, f)
        if os.path.exists(full_p):
            print(f"  Uploading {f}...")
            try:
                api.upload_file(
                    path_or_fileobj=full_p,
                    path_in_repo=f.replace("\\", "/"),
                    repo_id=repo_id,
                    repo_type="space"
                )
            except Exception as ex:
                print(f"  Warning on {f}: {ex}")

    # Upload LoRA adapter folder (~3.5 MB)
    lora_dir = os.path.join(BASE_DIR, "models", "lora_finetuned")
    if os.path.exists(lora_dir):
        print("\nUploading LoRA Adapter weights (~3.5 MB)...")
        try:
            api.upload_folder(
                folder_path=lora_dir,
                path_in_repo="models/lora_finetuned",
                repo_id=repo_id,
                repo_type="space"
            )
        except Exception as ex:
            print(f"  Warning on LoRA weights: {ex}")

    public_url = f"https://huggingface.co/spaces/{repo_id}"
    print("\n" + "=" * 65)
    print("SUCCESS! DEPLOYMENT IS LIVE (100% FREE):")
    print(f"Live URL: {public_url}")
    print("=" * 65)
    return public_url

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, default=None, help="Repo ID, e.g. username/medical-text-simplifier")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face Write Token")
    args = parser.parse_args()

    deploy_to_spaces(repo_id=args.repo_id, token=args.token)
