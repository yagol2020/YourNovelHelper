# YourNovelHelper

基于 Qwen3-7B 的小说风格微调项目，帮助用户创建自己喜欢风格的小说。

## 项目简介

YourNovelHelper 是一个小说创作辅助工具，通过收集小说文本数据集，使用 LoRA/QLoRA 技术微调 Qwen3-7B 模型，从而学习特定的小说风格，帮助用户更好地创作小说。

## 功能特性

- **数据预处理**: 将原始小说文本转换为训练数据
- **LoRA 微调**: 使用 QLoRA 低成本微调 Qwen3-7B 模型
- **命令行工具**: 交互式小说续写
- **Web API**: FastAPI 服务接口
- **Web UI**: Gradio 图形界面

## 项目结构

```
YourNovelHelper/
├── config/
│   └── config.yaml          # 配置文件
├── data/
│   ├── raw/                  # 原始数据
│   ├── processed/            # 处理后数据
│   └── output/               # 输出目录
├── src/
│   ├── data/
│   │   └── preprocess.py     # 数据预处理
│   ├── training/
│   │   └── train.py          # 模型训练
│   ├── inference/
│   │   └── generate.py       # 推理生成
│   └── api/
│       ├── main.py           # FastAPI 服务
│       └── webui.py          # Gradio Web UI
├── models/                   # 模型存储
├── logs/                     # 日志
└── scripts/                  # 脚本
```

## 快速开始

### 1. 创建虚拟环境并安装依赖

推荐使用 [uv](https://github.com/astral-sh/uv) 管理 Python 环境：

```bash
# 创建虚拟环境
uv venv

# 激活环境 (Linux/Mac)
source .venv/bin/activate

# Windows
# .venv\Scripts\activate

# 安装依赖
uv pip install torch transformers peft datasets trl accelerate pyyaml
uv pip install fastapi uvicorn gradio jieba tqdm scikit-learn
```

或者安装项目（包含所有依赖）：

```bash
uv pip install -e .
```

> **注意**: 如果没有 uv，请先安装: `pip install uv`

> 激活环境后，后续命令可以直接使用 `python` 运行。

### 2. 准备数据

将小说文本文件放入 `data/raw/` 目录，支持格式:
- `.txt` 文件
- `.json` 文件 (包含 `text` 字段或 `texts` 数组)

### 3. 数据预处理

```bash
python -m src.data.preprocess --raw-dir data/raw --output-dir data/processed
```

### 4. 训练模型

```bash
python -m src.training.train
```

训练参数可在 `config/config.yaml` 中修改。

### 5. 使用模型

#### 命令行

```bash
# 交互模式
python -m src.inference.generate --interactive

# 单次生成
python -m src.inference.generate --prompt "清晨的阳光透过窗户"
```

#### Web API

```bash
python -m src.api.main
```

访问 http://localhost:8000/docs 查看 API 文档。

#### Web UI

```bash
python -m src.api.webui
```

访问 http://localhost:7860 打开 Web 界面。

## 配置说明

`config/config.yaml` 主要配置项:

- `model.name`: 模型名称 (默认 Qwen/Qwen3-7B)
- `training.method`: 训练方法 (qlora)
- `training.lora_rank`: LoRA rank
- `training.num_epochs`: 训练轮数
- `inference.temperature`: 生成温度
- `api.port`: API 服务端口

## 硬件要求

- **训练**: 至少 16GB 显存的 GPU (QLoRA)
- **推理**: 至少 8GB 显存

## 示例

```python
from src.inference.generate import NovelGenerator

generator = NovelGenerator("models/novel-qlora")
result = generator.generate(
    prompt="雨夜，城市的一角",
    style_prompt="金庸的武侠风格"
)
print(result)
```

## API 示例

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "主人公走在街上",
    "style": "悬疑推理",
    "max_new_tokens": 1000
  }'
```

## 许可证

MIT License
