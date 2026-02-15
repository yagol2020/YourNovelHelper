FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/
COPY AGENTS.md /app/

RUN pip install --no-cache-dir \
    torch==2.5.1 \
    transformers>=4.40.0 \
    peft>=0.10.0 \
    datasets>=2.18.0 \
    trl>=0.8.0 \
    scikit-learn>=1.4.0 \
    accelerate>=0.28.0 \
    pyyaml>=6.0 \
    fastapi>=0.110.0 \
    uvicorn>=0.27.0 \
    gradio>=4.0.0 \
    jieba>=0.42.0 \
    tqdm>=4.66.0

COPY src /app/src
COPY config /app/config

RUN mkdir -p /app/data/raw /app/data/processed /app/models

ENV PYTHONPATH=/app:$PYTHONPATH

EXPOSE 8000

CMD ["python", "-m", "src.api.main"]
