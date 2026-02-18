# AGENTS.md - Developer Guidelines for YourNovelHelper

YourNovelHelper is a novel writing assistant based on LLM, using LoRA/QLoRA fine-tuning.
**This is a uv project** - always use uv for package management.

---

## 1. Build/Lint/Test Commands (uv)

```bash
# Setup - ALWAYS use uv
uv venv && source .venv/bin/activate
uv pip install torch transformers peft datasets trl accelerate pyyaml
uv pip install fastapi uvicorn gradio jieba tqdm scikit-learn
uv pip install -e .

# Run (use python module syntax)
python -m src.data.preprocess --raw-dir data/raw --output-dir data/processed
python -m src.training.train --config config/config.yaml
python -m src.api.main
python -m src.api.webui
python -m src.inference.generate --prompt "你的小说开头" --interactive

# Lint & Test
ruff check src/ && ruff format src/
mypy src/
pytest tests/ -v                    # All tests
pytest tests/test_preprocess.py -v  # Single file
pytest tests/test_preprocess.py::test_split_into_chunks -v  # Single test
pytest tests/ --cov=src --cov-report=term-missing  # Coverage
```

---

## 2. Code Style Guidelines

### General
- **No comments** unless explicitly requested
- **Type hints** required for all function arguments and return values
- **Error handling**: Use specific exception types, never bare `except:`

### Imports (order: stdlib → third-party → local)
```python
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import yaml
import torch
from transformers import AutoModelForCausalLM

from src.inference.generate import NovelGenerator
```

### Formatting
- Line length: 100 chars max, Indentation: 4 spaces
- Blank lines: Two between top-level definitions
- Quotes: Double quotes, single only when string contains double

### Naming Conventions
| Type        | Convention  | Example                  |
|-------------|-------------|--------------------------|
| Modules     | snake_case  | `preprocess.py`          |
| Classes     | PascalCase  | `NovelDatasetProcessor` |
| Functions   | snake_case  | `load_raw_texts`        |
| Variables   | snake_case  | `training_data`         |
| Constants   | UPPER_SNAKE | `MAX_SEQ_LENGTH`        |
| Dataclasses | PascalCase  | `class DataConfig:`     |

### Dataclasses for Configuration
```python
from dataclasses import dataclass, field

@dataclass
class DataConfig:
    raw_dir: str = "data/raw"
    chunk_size: int = 512
    target_modules: list = field(default_factory=lambda: ["q_proj", "k_proj"])
```

### Error Handling
```python
try:
    with open(file, "r", encoding="utf-8") as f:
        content = f.read()
except FileNotFoundError:
    print(f"Error: File {file} not found")
except Exception as e:
    print(f"Error reading {file}: {e}")
```

### Path Handling & JSONL
```python
output_path = Path(output_dir)
output_path.mkdir(parents=True, exist_ok=True)

# Reading JSONL
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)

# Writing JSONL
with open(file_path, "w", encoding="utf-8") as f:
    for item in data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
```

### Config Loading Pattern
```python
@dataclass
class TrainConfig:
    model_name: str = "Qwen2.5-0.5B"
    target_modules: list = field(default_factory=lambda: ["q_proj", "k_proj"])

def __init__(self, config_path: str = "config/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    self.train_config = TrainConfig()
    for key, value in config.get("training", {}).items():
        if hasattr(self.train_config, key):
            setattr(self.train_config, key, value)
```

### Environment Variables for Suppressing Logs
```python
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS"] = "1"
```

### Function/Class Design
- Functions under 50 lines, single responsibility
- Early returns for error conditions
- Explicit returns over implicit None
- `__init__` for initialization only
- Private methods prefixed with `_`

---

## 3. Project Structure
```
src/
├── data/         # Data preprocessing
├── training/     # Training scripts
├── inference/    # Generation/inference
└── api/         # FastAPI + Gradio UI
tests/           # Pytest tests
config/          # YAML configs
data/raw/        # Input data
data/processed/  # Processed output
```

---

## 4. Important Notes
- **uv always**: Use `uv pip install` NOT `pip install`
- **Always commit**: After every file modification, commit to git and push to GitHub
- **No proactive changes**: Get user approval before modifying code
- **Verify before commit**: Run `ruff check src/`
- **Security**: Never expose API keys or secrets
- **Chinese text**: Use `encoding="utf-8"` and `ensure_ascii=False` for JSON
