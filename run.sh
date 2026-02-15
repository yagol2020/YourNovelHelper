#!/bin/bash

# YourNovelHelper - 小说微调一键脚本

set -e

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== YourNovelHelper ===${NC}"

# 激活虚拟环境
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    uv venv && source .venv/bin/activate
fi

# 检查参数
MODE=${1:-help}

case $MODE in
    preprocess)
        echo -e "${GREEN}[1/3] 数据预处理${NC}"
        python -m src.data.preprocess --raw-dir data/raw --output-dir data/processed
        ;;

    train)
        echo -e "${GREEN}[2/3] 模型微调${NC}"
        python -m src.training.train --config config/config.yaml
        ;;

    api)
        echo -e "${GREEN}启动 API 服务${NC}"
        python -m src.api.main
        ;;

    webui)
        echo -e "${GREEN}启动 Web UI${NC}"
        python -m src.api.webui
        ;;

    generate)
        LORA_PATH="models/checkpoints/final"
        if [ -n "$2" ]; then
            echo -e "${GREEN}单次生成模式 (加载微调模型: $LORA_PATH)${NC}"
            python -m src.inference.generate --prompt "$2" --lora "$LORA_PATH"
        else
            echo -e "${GREEN}交互模式 (加载微调模型: $LORA_PATH)${NC}"
            python -m src.inference.generate --interactive --lora "$LORA_PATH"
        fi
        ;;

    base)
        if [ -n "$2" ]; then
            echo -e "${GREEN}单次生成模式 (基础模型)${NC}"
            python -m src.inference.generate --prompt "$2"
        else
            echo -e "${GREEN}交互模式 (基础模型)${NC}"
            python -m src.inference.generate --interactive
        fi
        ;;

    all)
        echo -e "${GREEN}执行完整流程: 预处理 -> 训练${NC}"
        echo -e "${GREEN}[1/2] 数据预处理${NC}"
        python -m src.data.preprocess --raw-dir data/raw --output-dir data/processed
        echo -e "${GREEN}[2/2] 模型微调${NC}"
        python -m src.training.train --config config/config.yaml
        ;;

    help|*)
        echo "用法: ./run.sh <command>"
        echo ""
        echo "命令:"
        echo "  preprocess  - 数据预处理"
        echo "  train       - 模型微调"
        echo "  api         - 启动 API 服务"
        echo "  webui       - 启动 Web UI"
        echo "  generate    - CLI 推理模式 (使用微调模型)"
        echo "  base        - CLI 推理模式 (使用基础模型)"
        echo "  all         - 完整流程 (预处理 + 训练)"
        echo "  help        - 显示帮助"
        ;;
esac
