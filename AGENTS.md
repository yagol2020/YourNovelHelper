# AGENTS.md - Developer Guidelines for YourNovelHelper

YourNovelHelper is a novel writing assistant based on Qwen3-7B, using LoRA/QLoRA fine-tuning. The project consists of data preprocessing, training, inference, API, and WebUI components.

---

## 1. Build/Lint/Test Commands

```bash
# Activate virtual environment
source .venv/bin/activate

# Data preprocessing
python -m src.data.preprocess --raw-dir data/raw --output-dir data/processed

# Training
python -m src.training.train --config config/config.yaml

# API server
python -m src.api.main

# Web UI
python -m src.api.webui

# CLI inference
python -m src.inference.generate --prompt "你的小说开头" --interactive

# Linting (ruff)
ruff check src/
ruff check src/ --fix
ruff format src/

# Type checking
mypy src/

# Tests (pytest)
pytest tests/test_preprocess.py -v
pytest tests/test_preprocess.py::test_split_into_chunks -v
pytest tests/ --cov=src --cov-report=term-missing
```

---

## 2. Code Style Guidelines

### General Principles
- **No comments**: Do NOT add comments unless explicitly requested
- **Type hints**: Use type hints for all function arguments and return values
- **Error handling**: Use try-except with specific exception types, not bare except

### Imports
```python
# Standard library first
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

# Third-party
import yaml
import torch
from transformers import AutoModelForCausalLM

# Local - use absolute imports
from src.inference.generate import NovelGenerator
```

### Formatting
- **Line length**: Maximum 100 characters
- **Indentation**: 4 spaces
- **Blank lines**: Two between top-level definitions
- **Quotes**: Double quotes, single only when string contains double quotes

### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Modules | snake_case | `preprocess.py` |
| Classes | PascalCase | `NovelDatasetProcessor` |
| Functions | snake_case | `load_raw_texts` |
| Variables | snake_case | `training_data` |
| Constants | UPPER_SNAKE | `MAX_SEQ_LENGTH` |
| Dataclasses | PascalCase | `class DataConfig:` |

### Type Annotations
```python
# Good
def process(self, raw_dir: str = "data/raw") -> Dict[str, List[Any]]:
    ...

# Avoid
def process(self, raw_dir="data/raw"):  # No type hints
    ...
```

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
# Good
try:
    with open(file, "r", encoding="utf-8") as f:
        content = f.read()
except FileNotFoundError:
    print(f"Error: File {file} not found")
except Exception as e:
    print(f"Error reading {file}: {e}")

# Never do this
try:
    ...
except:  # BAD
    ...
```

### Function Design
- Keep functions under 50 lines
- Single responsibility
- Use early returns for error conditions
- Prefer explicit returns over implicit None

```python
def load_texts(self, path: str) -> List[str]:
    if not Path(path).exists():
        return []
    
    texts = []
    for file in Path(path).glob("*.txt"):
        texts.append(self._read_file(file))
    return texts
```

### Class Design
- Use `__init__` for initialization only
- Use private methods (prefixed with `_`) for internal logic

```python
class NovelDatasetProcessor:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
    
    def process(self, raw_dir: str, output_dir: str) -> None:
        ...
    
    def _load_raw_texts(self) -> List[str]:
        ...
```

---

## 3. Common Patterns

### Path Handling
```python
from pathlib import Path

output_path = Path(output_dir)
output_path.mkdir(parents=True, exist_ok=True)

if Path(path).exists():
    ...
```

### Configuration Loading
```python
import yaml
from dataclasses import dataclass

@dataclass
class TrainConfig:
    model_name: str = "Qwen/Qwen3-7B"

def __init__(self, config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    self.train_config = TrainConfig()
    for key, value in config.get("training", {}).items():
        if hasattr(self.train_config, key):
            setattr(self.train_config, key, value)
```

### JSONL File Operations
```python
# Reading
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        ...

# Writing
with open(file_path, "w", encoding="utf-8") as f:
    for item in data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
```

---

## 4. Important Notes

- **No proactive changes**: Do NOT make changes without user approval
- **Verify before committing**: Always run linting before suggesting commits
- **Security**: Never expose API keys or secrets in code
- **GPU memory**: Be mindful of memory usage when loading large models
