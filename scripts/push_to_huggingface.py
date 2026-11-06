#!/usr/bin/env python3
"""Automated Publishing Script for Hugging Face Models and Spaces.

Usage:
    export HF_TOKEN="hf_your_token_here"
    python scripts/push_to_huggingface.py --username shubhamkumar12864
"""
import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

ROOT = Path(__file__).resolve().parents[1]
EXPORT_DIR = ROOT / "reports" / "unified_extended" / "export"
DEPLOY_DIR = ROOT / "deploy"


def publish_to_huggingface(username: str, token: str):
    api = HfApi(token=token)
    model_repo_id = f"{username}/plant-leaf-disease-detection-using-cnn"
    space_repo_id = f"{username}/plant-leaf-disease-detection-using-cnn"

    print("\n=======================================================")
    print("🚀 PUBLISHING RESEARCH ASSETS TO HUGGING FACE")
    print("=======================================================")
    print(f"Target Username:    {username}")
    print(f"Model Repository:   https://huggingface.co/{model_repo_id}")
    print(f"Interactive Space:  https://huggingface.co/spaces/{space_repo_id}")

    # 1. Create & Upload Model Repository
    print(f"\n[1/2] Syncing Model Repository ({model_repo_id})...")
    create_repo(repo_id=model_repo_id, repo_type="model", token=token, exist_ok=True)

    # Prepare model staging directory
    model_stage = DEPLOY_DIR / "hf_model"
    model_stage.mkdir(parents=True, exist_ok=True)

    # Link/copy export artifacts
    onnx_fp32 = EXPORT_DIR / "unified_gated_plant_cnn.onnx"
    onnx_int8 = EXPORT_DIR / "unified_gated_plant_cnn_int8.onnx"
    pt_best = ROOT / "reports" / "unified_extended" / "unified_gated_plant_cnn_extended_best.pt"
    calib_json = ROOT / "reports" / "unified_extended" / "conformal_calibration_results.json"
    bench_json = EXPORT_DIR / "edge_benchmark_summary.json"

    files_to_upload = [
        (model_stage / "README.md", "README.md"),
        (onnx_int8, "unified_gated_plant_cnn_int8.onnx"),
        (onnx_fp32, "unified_gated_plant_cnn.onnx"),
        (pt_best, "unified_gated_plant_cnn_extended_best.pt"),
        (EXPORT_DIR / "edge_latency_benchmark.csv", "edge_latency_benchmark.csv"),
        (bench_json, "edge_benchmark_summary.json"),
        (calib_json, "conformal_calibration_results.json"),
    ]

    for src, dst_name in files_to_upload:
        if src.exists():
            print(f"  Uploading {src.name} -> {dst_name}...")
            api.upload_file(
                path_or_fileobj=str(src),
                path_in_repo=dst_name,
                repo_id=model_repo_id,
                repo_type="model",
                commit_message=f"Upload {dst_name} [scientific release]"
            )

    print("  ✅ Model repository published successfully!")

    # 2. Create & Upload Space Repository (Clean In-Browser WebAssembly Edge App)
    print(f"\n[2/2] Syncing Interactive Space ({space_repo_id})...")
    create_repo(
        repo_id=space_repo_id,
        repo_type="space",
        space_sdk="static",
        token=token,
        exist_ok=True
    )

    # Clean old python or obsolete files if present
    obsolete_files = [
        "app.py",
        "requirements.txt",
        "unified_gated_plant_cnn_int8.onnx",
        "style.css",
    ]
    for obs in obsolete_files:
        try:
            api.delete_file(
                path_in_repo=obs,
                repo_id=space_repo_id,
                repo_type="space",
                commit_message=f"Clean {obs}"
            )
        except Exception:
            pass

    # Clean old src folder if present
    try:
        api.delete_folder(
            path_in_repo="src",
            repo_id=space_repo_id,
            repo_type="space",
            commit_message="Clean src folder"
        )
    except Exception:
        pass

    space_dir = DEPLOY_DIR / "hf_static_space"

    print(f"  Uploading clean Web App from {space_dir}...")
    api.upload_folder(
        folder_path=str(space_dir),
        repo_id=space_repo_id,
        repo_type="space",
        commit_message="Deploy clean in-browser WebAssembly plant leaf disease diagnostic app",
        ignore_patterns=["*__pycache__*", "*.pyc", ".pytest_cache*"]
    )

    print("\n=======================================================")
    print("🎉 DEPLOYMENT COMPLETE!")
    print(f"👉 Public Model: https://huggingface.co/{model_repo_id}")
    print(f"👉 Live Web App: https://huggingface.co/spaces/{space_repo_id}")
    print("=======================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Publish Plant CNN to Hugging Face")
    parser.add_argument("--username", type=str, default=None, help="Hugging Face username")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face User Access Token (write permissions)")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("\n[Notice] No Hugging Face token provided.")
        print("To push to Hugging Face, generate a free token at https://huggingface.co/settings/tokens (Write scope), then run:")
        print(f"  export HF_TOKEN=\"hf_your_token_here\"")
        print(f"  python scripts/push_to_huggingface.py\n")
        return

    api = HfApi(token=token)
    try:
        user_info = api.whoami()
        username = args.username or user_info.get("name")
    except Exception as e:
        print(f"Error fetching user info from Hugging Face: {e}")
        username = args.username or "Shubham12864"

    publish_to_huggingface(username, token)


if __name__ == "__main__":
    main()
