"""
FastAPI 服务模块

提供 RESTful API 接口用于小说生成，支持：
- 文本续写生成
- 可配置的生成参数
- 健康检查
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.inference.generate import NovelGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


# 创建 FastAPI 应用
app = FastAPI(title="YourNovelHelper API", version="1.0.0")

# 配置 CORS 中间件，允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 请求体模型
class GenerateRequest(BaseModel):
    """生成请求参数"""

    prompt: str  # 续写开头
    style: Optional[str] = ""  # 风格描述
    max_new_tokens: Optional[int] = 2048  # 最大生成长度
    temperature: Optional[float] = 0.7  # 温度参数
    top_p: Optional[float] = 0.8  # top_p 采样
    top_k: Optional[int] = 20  # top_k 采样
    repetition_penalty: Optional[float] = 1.1  # 重复惩罚


# 响应体模型
class GenerateResponse(BaseModel):
    """生成响应参数"""

    result: str  # 生成的文本
    model: str  # 模型路径


# 全局生成器实例
generator = None


@app.on_event("startup")
async def startup_event():
    """
    应用启动时的事件处理器

    尝试加载模型，如果失败不影响 API 启动（只是生成接口不可用）
    """
    global generator
    config_path = "config/config.yaml"
    model_path = "models/novel-qlora"

    try:
        generator = NovelGenerator(model_path, config_path)
        generator.load_model()
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        print(
            "API will work but generation endpoints require model to be trained first"
        )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    文本生成接口

    根据提供的 prompt 生成小说续写内容。

    Args:
        request: 生成请求参数

    Returns:
        包含生成结果的响应

    Raises:
        HTTPException: 模型未加载时返回 503 错误
    """
    if generator is None or generator.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # 保存原有参数
    old_max_tokens = generator.max_new_tokens
    old_temp = generator.temperature
    old_top_p = generator.top_p
    old_top_k = generator.top_k
    old_rep_pen = generator.repetition_penalty

    # 应用请求中的参数
    generator.max_new_tokens = request.max_new_tokens
    generator.temperature = request.temperature
    generator.top_p = request.top_p
    generator.top_k = request.top_k
    generator.repetition_penalty = request.repetition_penalty

    try:
        result = generator.generate(request.prompt, request.style)
    finally:
        # 恢复原有参数，避免影响其他请求
        generator.max_new_tokens = old_max_tokens
        generator.temperature = old_temp
        generator.top_p = old_top_p
        generator.top_k = old_top_k
        generator.repetition_penalty = old_rep_pen

    return GenerateResponse(result=result, model=generator.model_path)


@app.get("/health")
async def health():
    """
    健康检查接口

    Returns:
        服务状态和模型加载状态
    """
    return {
        "status": "healthy",
        "model_loaded": generator is not None and generator.model is not None,
    }


@app.get("/")
async def root():
    """
    根路径接口

    Returns:
        服务基本信息
    """
    return {
        "message": "YourNovelHelper API",
        "docs": "/docs",
        "endpoints": ["/generate", "/health"],
    }


if __name__ == "__main__":
    import uvicorn
    import yaml

    # 加载配置并启动服务
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    api_config = config.get("api", {})
    uvicorn.run(
        app,
        host=api_config.get("host", "0.0.0.0"),
        port=api_config.get("port", 8000),
        workers=api_config.get("workers", 1),
    )
