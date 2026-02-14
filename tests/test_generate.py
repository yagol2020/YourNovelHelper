import pytest
import tempfile
from pathlib import Path
import sys
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.generate import NovelGenerator


@pytest.fixture
def temp_config_file():
    config_data = {
        "inference": {
            "max_new_tokens": 2048,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "repetition_penalty": 1.1,
        }
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        temp_path = f.name

    yield temp_path

    Path(temp_path).unlink(missing_ok=True)


def test_generator_init_defaults():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump({"inference": {}}, f)
        temp_path = f.name

    try:
        generator = NovelGenerator(config_path=temp_path)
        assert generator.max_new_tokens == 2048
        assert generator.temperature == 0.7
        assert generator.top_p == 0.8
    finally:
        Path(temp_path).unlink()


def test_generator_init_with_config(temp_config_file):
    generator = NovelGenerator(config_path=temp_config_file)
    assert generator.max_new_tokens == 2048
    assert generator.temperature == 0.7
    assert generator.top_p == 0.8
    assert generator.top_k == 20
    assert generator.repetition_penalty == 1.1


def test_generator_init_custom_model():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump({"inference": {}}, f)
        temp_path = f.name

    try:
        generator = NovelGenerator(
            model_path="Qwen3-4B", lora_path="models/test-lora", config_path=temp_path
        )
        assert generator.model_path == "Qwen3-4B"
        assert generator.lora_path == "models/test-lora"
    finally:
        Path(temp_path).unlink()


def test_generator_model_not_loaded_initially(temp_config_file):
    generator = NovelGenerator(config_path=temp_config_file)
    assert generator.model is None
    assert generator.tokenizer is None


def test_generate_params_override():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(
            {
                "inference": {
                    "max_new_tokens": 100,
                    "temperature": 0.5,
                }
            },
            f,
        )
        temp_path = f.name

    try:
        generator = NovelGenerator(config_path=temp_path)
        assert generator.max_new_tokens == 100
        assert generator.temperature == 0.5

        generator.max_new_tokens = 500
        generator.temperature = 0.9

        assert generator.max_new_tokens == 500
        assert generator.temperature == 0.9
    finally:
        Path(temp_path).unlink()
