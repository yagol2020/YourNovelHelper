# Docker 部署指南

## 前置要求

- Docker >= 20.10
- Docker Compose >= 2.0
- NVIDIA GPU + nvidia-docker2

## 目录结构

部署前请确保以下目录存在：

```
YourNovelHelper/
├── Dockerfile
├── docker-compose.yml
├── src/
├── config/
├── data/
│   └── raw/           # 小说源文件
├── models/            # 模型目录
│   ├── Qwen3-4B/      # 基础模型
│   └── checkpoints/  # LoRA 微调权重
└── .env (可选)
```

## 准备模型文件

1. **下载基础模型** 到 `models/Qwen3-4B/` 目录
2. **复制微调后的 LoRA 权重** 到 `models/checkpoints/` 目录

## 配置

编辑 `.env` 文件（从 `.env.example` 复制）:

```bash
MODEL_PATH=models/Qwen3-4B
LORA_PATH=models/checkpoints/checkpoint-3
CONFIG_PATH=config/config.yaml
```

## 构建镜像

```bash
docker compose build
```

## 运行服务

### API 服务 (端口 8000)

```bash
docker compose up novel-api
```

### WebUI (端口 7860)

```bash
docker compose up novel-webui
```

### 后台运行

```bash
docker compose up -d
```

## 查看日志

```bash
docker compose logs -f novel-api
```

## 停止服务

```bash
docker compose down
```

## API 使用示例

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "主角走在森林里，",
    "max_new_tokens": 500,
    "temperature": 0.7
  }'
```

## 访问

- API 文档: http://localhost:8000/docs
- WebUI: http://localhost:7860
