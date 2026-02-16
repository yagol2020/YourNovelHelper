FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    wget \
    vim \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

COPY pyproject.toml /app/

RUN uv venv /app/.venv && \
    . /app/.venv/bin/activate && \
    uv pip install --system -e .

COPY src /app/src
COPY config /app/config
COPY scripts /app/scripts

RUN mkdir -p /app/data/raw /app/data/processed /app/models

ENV PYTHONPATH=/app:$PYTHONPATH
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["python", "-m", "src.api.main"]
