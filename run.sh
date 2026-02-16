#!/bin/bash

set -e

CONTAINER_NAME="your-novel-helper"
IMAGE_NAME="your-novel-helper:latest"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m'

init_python_env() {
    cd "$SCRIPT_DIR"
    
    if [ -d ".venv" ] && [ -f ".venv/bin/python" ]; then
        PYTHON_CMD="$SCRIPT_DIR/.venv/bin/python"
        echo -e "${GREEN}Using existing virtual environment${NC}"
        return 0
    fi
    
    if command -v uv &> /dev/null; then
        echo -e "${YELLOW}Creating virtual environment with uv...${NC}"
        uv venv
        uv pip install -e .
        PYTHON_CMD="$SCRIPT_DIR/.venv/bin/python"
        echo -e "${GREEN}Virtual environment created successfully${NC}"
        return 0
    fi
    
    if command -v python3 &> /dev/null; then
        echo -e "${YELLOW}uv not found, using system python3${NC}"
        PYTHON_CMD="python3"
        return 0
    fi
    
    echo -e "${YELLOW}Error: Neither uv nor python3 found${NC}"
    echo "Please install uv first: pip install uv"
    exit 1
}

show_usage() {
    cat << EOF
YourNovelHelper - 统一运行入口

用法: $0 <cli|docker> <command> [options]

Arguments:
  cli|docker        运行模式：cli=宿主机, docker=容器内

Commands:
  download    从 ModelScope 下载模型
              参数:
                --model    模型名称 (默认: Qwen3-4B)
                --output   输出目录 (默认: models)
              示例: $0 cli download --model Qwen3-4B --output ./models
  
  preprocess  预处理小说数据集
              参数:
                --raw-dir     原始数据目录 (默认: data/raw)
                --output-dir  输出目录 (默认: data/processed)
                --config      配置文件 (默认: config/config.yaml)
              示例: $0 cli preprocess --raw-dir ./data/raw --output-dir ./data/processed
  
  train       微调模型
              参数:
                --model-dir   模型目录 (默认: models)
                --data-dir    训练数据目录 (默认: data/processed)
                --output-dir  输出目录 (默认: models/checkpoints)
                --logs-dir    日志目录 (默认: logs)
                --config      配置文件 (默认: config/config.yaml)
              示例: $0 cli train --model-dir ./models --data-dir ./data/processed --output-dir ./models/checkpoints --logs-dir ./logs
  
  generate    文本补全
              参数:
                --model-dir   模型目录 (默认: models)
                --base-model  基础模型名称 (默认: Qwen3-4B)
                --lora        LoRA 模型路径 (可选)
                --prompt      补全的文本 (无此参数进入交互模式)
                -i, --interactive 交互模式
              示例: 
                $0 cli generate --model-dir ./models --base-model Qwen3-4B --lora ./models/novel-qlora --prompt "小说开头"
                $0 cli generate --model-dir ./models --base-model Qwen3-4B --interactive

Examples:
  # CLI 模式
  $0 cli download --model Qwen3-4B --output ./models
  $0 cli preprocess --raw-dir ./data/raw --output-dir ./data/processed
  $0 cli train --model-dir ./models --data-dir ./data/processed --output-dir ./models/checkpoints --logs-dir ./logs
  $0 cli generate --model-dir ./models --base-model Qwen3-4B --lora ./models/novel-qlora --prompt "开头"
  
  # Docker 模式
  $0 docker download --model Qwen3-4B --output ./models
  $0 docker preprocess --raw-dir ./data/raw --output-dir ./data/processed
  $0 docker train --model-dir ./models --data-dir ./data/processed --output-dir ./models/checkpoints --logs-dir ./logs
  $0 docker generate --model-dir ./models --base-model Qwen3-4B --prompt "开头"

EOF
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        echo "Error: docker not found"
        exit 1
    fi
}

build_container() {
    echo "Building Docker image..."
    docker build -t "$IMAGE_NAME" "$SCRIPT_DIR"
}

start_container() {
    local model_dir="$1"
    local data_dir="$2"
    
    if [ -z "$model_dir" ]; then
        model_dir="./models"
    fi
    if [ -z "$data_dir" ]; then
        data_dir="./data"
    fi
    
    mkdir -p "$model_dir" "$data_dir/raw" "$data_dir/processed"
    
    CONTAINER_ID=$(docker ps -aq -f name="^${CONTAINER_NAME}$")
    if [ -z "$CONTAINER_ID" ]; then
        docker run -d --name "$CONTAINER_NAME" \
            -v "$(pwd)/$model_dir:/app/models" \
            -v "$(pwd)/$data_dir:/app/data" \
            -v "$(pwd)/config:/app/config" \
            -v "$(pwd)/scripts:/app/scripts" \
            "$IMAGE_NAME" \
            tail -f /dev/null
    fi
}

ensure_container() {
    check_docker
    
    CONTAINER_ID=$(docker ps -aq -f name="^${CONTAINER_NAME}$")
    
    if [ -z "$CONTAINER_ID" ]; then
        build_container
        start_container
    fi
}

convert_to_container_path() {
    local path="$1"
    if [[ "$path" == /* ]]; then
        echo "$path"
    else
        echo "/app/$path"
    fi
}

cmd_download() {
    local model="Qwen3-4B"
    local output="models"
    local is_docker=0
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --model)
                model="$2"
                shift 2
                ;;
            --output)
                output="$2"
                shift 2
                ;;
            docker)
                is_docker=1
                shift
                ;;
            *)
                shift
                ;;
        esac
    done
    
    if [ $is_docker -eq 1 ]; then
        ensure_container
        local container_output=$(convert_to_container_path "$output")
        docker exec "$CONTAINER_NAME" python scripts/download_model.py --model "$model" --output "$container_output"
    else
        $PYTHON_CMD scripts/download_model.py --model "$model" --output "$output"
    fi
}

cmd_preprocess() {
    local raw_dir="data/raw"
    local output_dir="data/processed"
    local config="config/config.yaml"
    local is_docker=0
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --raw-dir)
                raw_dir="$2"
                shift 2
                ;;
            --output-dir)
                output_dir="$2"
                shift 2
                ;;
            --config)
                config="$2"
                shift 2
                ;;
            docker)
                is_docker=1
                shift
                ;;
            *)
                shift
                ;;
        esac
    done
    
    if [ $is_docker -eq 1 ]; then
        ensure_container
        local container_raw=$(convert_to_container_path "$raw_dir")
        local container_output=$(convert_to_container_path "$output_dir")
        local container_config=$(convert_to_container_path "$config")
        docker exec "$CONTAINER_NAME" python -m src.data.preprocess --raw-dir "$container_raw" --output-dir "$container_output" --config "$container_config"
    else
        $PYTHON_CMD -m src.data.preprocess --raw-dir "$raw_dir" --output-dir "$output_dir" --config "$config"
    fi
}

cmd_train() {
    local model_dir="models"
    local data_dir="data/processed"
    local output_dir="models/checkpoints"
    local logs_dir="logs"
    local config="config/config.yaml"
    local is_docker=0
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --model-dir)
                model_dir="$2"
                shift 2
                ;;
            --data-dir)
                data_dir="$2"
                shift 2
                ;;
            --output-dir)
                output_dir="$2"
                shift 2
                ;;
            --logs-dir)
                logs_dir="$2"
                shift 2
                ;;
            --config)
                config="$2"
                shift 2
                ;;
            docker)
                is_docker=1
                shift
                ;;
            *)
                shift
                ;;
        esac
    done
    
    export TENSORBOARD_LOG_DIR="$logs_dir"
    
    if [ $is_docker -eq 1 ]; then
        ensure_container
        local container_model=$(convert_to_container_path "$model_dir")
        local container_data=$(convert_to_container_path "$data_dir")
        local container_output=$(convert_to_container_path "$output_dir")
        local container_logs=$(convert_to_container_path "$logs_dir")
        local container_config=$(convert_to_container_path "$config")
        
        mkdir -p "$(pwd)/$logs_dir"
        
        docker exec -e TENSORBOARD_LOG_DIR="$container_logs" -e TENSORBOARD_LOGGING_DIR="$container_logs" "$CONTAINER_NAME" \
            python -m src.training.train --config "$container_config"
    else
        mkdir -p "$output_dir" "$logs_dir"
        $PYTHON_CMD -m src.training.train --config "$config"
    fi
}

cmd_generate() {
    local model_dir="models"
    local base_model="Qwen3-4B"
    local lora=""
    local prompt=""
    local interactive=0
    local is_docker=0
    local extra_args=()
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --model-dir)
                model_dir="$2"
                shift 2
                ;;
            --base-model)
                base_model="$2"
                shift 2
                ;;
            --lora)
                lora="$2"
                shift 2
                ;;
            --prompt)
                prompt="$2"
                shift 2
                ;;
            -i|--interactive)
                interactive=1
                shift
                ;;
            docker)
                is_docker=1
                shift
                ;;
            *)
                shift
                ;;
        esac
    done
    
    local model_dir_arg="--model-dir $model_dir"
    local base_model_arg="--base-model $base_model"
    local lora_arg=""
    if [ -n "$lora" ]; then
        lora_arg="--lora $lora"
    fi
    
    if [ $is_docker -eq 1 ]; then
        ensure_container
        local container_model=$(convert_to_container_path "$model_dir")
        local container_lora=""
        if [ -n "$lora" ]; then
            container_lora=$(convert_to_container_path "$lora")
        fi
        
        if [ $interactive -eq 1 ]; then
            docker exec "$CONTAINER_NAME" python -m src.inference.generate \
                --model-dir "$container_model" \
                --base-model "$base_model" \
                $([ -n "$container_lora" ] && echo "--lora $container_lora") \
                --interactive
        else
            docker exec "$CONTAINER_NAME" python -m src.inference.generate \
                --model-dir "$container_model" \
                --base-model "$base_model" \
                $([ -n "$container_lora" ] && echo "--lora $container_lora") \
                --prompt "$prompt"
        fi
    else
        if [ $interactive -eq 1 ]; then
            $PYTHON_CMD -m src.inference.generate \
                $model_dir_arg \
                $base_model_arg \
                $([ -n "$lora_arg" ] && echo "$lora_arg") \
                --interactive
        else
            $PYTHON_CMD -m src.inference.generate \
                $model_dir_arg \
                $base_model_arg \
                $([ -n "$lora_arg" ] && echo "$lora_arg") \
                --prompt "$prompt"
        fi
    fi
}

if [ $# -lt 2 ]; then
    show_usage
    exit 1
fi

MODE="$1"
shift

if [ "$MODE" = "cli" ]; then
    init_python_env
fi

if [ "$MODE" != "cli" ] && [ "$MODE" != "docker" ]; then
    echo "Error: Invalid mode '$MODE'. Use 'cli' or 'docker'"
    show_usage
    exit 1
fi

COMMAND="$1"
shift

case "$COMMAND" in
    download)
        cmd_download "$@"
        ;;
    preprocess)
        cmd_preprocess "$@"
        ;;
    train)
        cmd_train "$@"
        ;;
    generate)
        cmd_generate "$@"
        ;;
    *)
        echo "Error: Unknown command '$COMMAND'"
        show_usage
        exit 1
        ;;
esac
