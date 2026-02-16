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

# 询问是否使用现有环境
echo "请选择环境配置方式:"
echo "  1) 创建新的虚拟环境 (推荐)"
echo "  2) 使用当前Python环境"
read -p "请输入选项 [1]: " ENV_OPTION
ENV_OPTION=${ENV_OPTION:-1}

if [ "$ENV_OPTION" == "1" ]; then
    echo ""
    # 询问是否安装uv
    if command -v uv &> /dev/null; then
        echo "检测到已安装 uv ✓"
        USE_UV=true
    else
        echo ""
        echo "是否安装 uv (现代化的Python包管理工具，比pip更快)?"
        echo "  1) 安装 uv (推荐)"
        echo "  2) 使用传统的 venv"
        read -p "请输入选项 [1]: " UV_OPTION
        UV_OPTION=${UV_OPTION:-1}
        
        if [ "$UV_OPTION" == "1" ]; then
            echo ""
            echo "Installing uv..."
            curl -LsSf https://astral.sh/uv/install.sh | sh
            USE_UV=true
        else
            USE_UV=false
        fi
    fi
    
    echo ""
    echo "创建虚拟环境..."
    if [ "$USE_UV" == "true" ]; then
        uv venv .venv
    else
        python3 -m venv .venv
    fi
    source .venv/bin/activate
    echo "虚拟环境创建完成 ✓"
else
    echo "使用当前环境"
    echo "建议先创建虚拟环境: python3 -m venv .venv && source .venv/bin/activate"
    echo ""
fi

echo ""
echo "Installing dependencies..."

if [ "$USE_UV" == "true" ]; then
    uv pip install torch transformers peft datasets trl accelerate pyyaml
    uv pip install fastapi uvicorn gradio jieba tqdm scikit-learn
    uv pip install 'setuptools<82'
else
    pip install torch transformers peft datasets trl accelerate pyyaml
    pip install fastapi uvicorn gradio jieba tqdm scikit-learn
    pip install 'setuptools<82'
fi

echo ""
echo "=== 安装完成 ==="
echo ""
echo "激活环境: source .venv/bin/activate"
echo "运行API:  python -m src.api.main"
echo "运行WebUI: python -m src.api.webui"
