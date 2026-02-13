from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.inference.generate import NovelGenerator


app = FastAPI(title="YourNovelHelper API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    prompt: str
    style: Optional[str] = ""
    max_new_tokens: Optional[int] = 2048
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.8
    top_k: Optional[int] = 20
    repetition_penalty: Optional[float] = 1.1


class GenerateResponse(BaseModel):
    result: str
    model: str


generator = None


@app.on_event("startup")
async def startup_event():
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
    if generator is None or generator.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    old_max_tokens = generator.max_new_tokens
    old_temp = generator.temperature
    old_top_p = generator.top_p
    old_top_k = generator.top_k
    old_rep_pen = generator.repetition_penalty

    generator.max_new_tokens = request.max_new_tokens
    generator.temperature = request.temperature
    generator.top_p = request.top_p
    generator.top_k = request.top_k
    generator.repetition_penalty = request.repetition_penalty

    try:
        result = generator.generate(request.prompt, request.style)
    finally:
        generator.max_new_tokens = old_max_tokens
        generator.temperature = old_temp
        generator.top_p = old_top_p
        generator.top_k = old_top_k
        generator.repetition_penalty = old_rep_pen

    return GenerateResponse(result=result, model=generator.model_path)


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": generator is not None and generator.model is not None,
    }


@app.get("/")
async def root():
    return {
        "message": "YourNovelHelper API",
        "docs": "/docs",
        "endpoints": ["/generate", "/health"],
    }


if __name__ == "__main__":
    import uvicorn
    import yaml

    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    api_config = config.get("api", {})
    uvicorn.run(
        app,
        host=api_config.get("host", "0.0.0.0"),
        port=api_config.get("port", 8000),
        workers=api_config.get("workers", 1),
    )
