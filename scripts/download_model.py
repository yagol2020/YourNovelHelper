#!/usr/bin/env python3
"""
下载 Qwen3 模型到本地目录

用法:
    python scripts/download_model.py              # 下载 Qwen3-4B (默认)
    python scripts/download_model.py --model Qwen3-7B  # 下载 Qwen3-7B
    python scripts/download_model.py --output ./models  # 指定输出目录
"""

import argparse
import os
import sys
from pathlib import Path

import torch

try:
    from modelscope import snapshot_download

    MODELSCOPE_AVAILABLE = True
except ImportError:
    MODELSCOPE_AVAILABLE = False


MODEL_CONFIGS = {
    "Qwen3-4B": {
        "model_id": "qwen/Qwen3-4B",
        "size": "~8GB",
    },
    "Qwen3-7B": {
        "model_id": "qwen/Qwen3-7B",
        "size": "~14GB",
    },
    "Qwen2.5-0.5B": {
        "model_id": "qwen/Qwen2.5-0.5B",
        "size": "~1GB",
    },
}


def download_model(model_name: str, output_dir: str) -> Path:
    if not MODELSCOPE_AVAILABLE:
        print("Error: modelscope not installed.")
        print("Please run: pip install modelscope")
        sys.exit(1)

    if model_name not in MODEL_CONFIGS:
        print(f"Error: Unknown model '{model_name}'")
        print(f"Available models: {', '.join(MODEL_CONFIGS.keys())}")
        sys.exit(1)

    config = MODEL_CONFIGS[model_name]
    model_id = config["model_id"]

    output_path = Path(output_dir) / model_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {model_name} ({config['size']}) to {output_path}")
    print(f"Model ID: {model_id}")
    print("This may take a while depending on your network speed...")

    try:
        local_dir = snapshot_download(
            model_id,
            cache_dir=output_dir,
            revision="master",
        )
        print(f"\nModel downloaded successfully!")
        print(f"Path: {local_dir}")

        target_link = output_path
        if Path(local_dir) != target_link and Path(local_dir).exists():
            if target_link.exists() or target_link.is_symlink():
                target_link.unlink()
            Path(local_dir).rename(target_link)
            print(f"Moved to: {target_link}")

        return target_link

    except Exception as e:
        print(f"Error downloading model: {e}")
        sys.exit(1)


def list_models():
    """列出可下载的模型"""
    print("\nAvailable models:")
    print("-" * 40)
    for name, config in MODEL_CONFIGS.items():
        print(f"  {name:20s} {config['size']:>10s}")
    print("-" * 40)


def main():
    parser = argparse.ArgumentParser(description="Download Qwen3 model from ModelScope")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen3-4B",
        choices=list(MODEL_CONFIGS.keys()),
        help="Model to download",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models",
        help="Output directory (default: models)",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List available models",
    )

    args = parser.parse_args()

    if args.list:
        list_models()
        return

    print("=" * 50)
    print("Qwen3 Model Downloader")
    print("=" * 50)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 50)

    download_model(args.model, args.output)

    print("\n" + "=" * 50)
    print("Usage:")
    print("=" * 50)
    print(f"Model saved to: {args.output}/{args.model}")
    print("\nCLI usage:")
    print(
        f"  python -m src.inference.generate --model-dir {args.output} --base-model {args.model} --prompt 'test'"
    )
    print("\nDocker usage:")
    print(f"  docker run -d -p 8000:8000 \\")
    print(f"    -v $(pwd)/{args.output}:/app/{args.output} \\")
    print(f"    -e MODEL_DIR=/app/{args.output} \\")
    print(f"    -e BASE_MODEL={args.model} \\")
    print(f"    your-novel-helper:latest")
    print("=" * 50)


if __name__ == "__main__":
    main()
