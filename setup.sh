#!/bin/bash

set -e

echo "=== YourNovelHelper 环境配置脚本 ==="
echo ""

# 检查Python版本
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
REQUIRED_VERSION="3.10"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "Error: Python 3.10+ is required, but found $PYTHON_VERSION"
    exit 1
fi

echo "Python version: $PYTHON_VERSION ✓"
echo ""

# 检查uv是否安装
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env" 2>/dev/null || source "$HOME/.local/uv/uv.sh" 2>/dev/null || true
fi

echo "uv installed ✓"
echo ""

# 创建虚拟环境
echo "Creating virtual environment..."
uv venv .venv
source .venv/bin/activate

echo "Virtual environment created ✓"
echo ""

# 安装依赖
echo "Installing dependencies..."
uv pip install torch transformers peft datasets trl accelerate pyyaml
uv pip install fastapi uvicorn gradio jieba tqdm scikit-learn

echo ""
echo "=== 安装完成 ==="
echo ""
echo "激活环境: source .venv/bin/activate"
echo "运行API:  python -m src.api.main"
echo "运行WebUI: python -m src.api.webui"
