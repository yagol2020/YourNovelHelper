import pytest
import tempfile
from pathlib import Path
import sys
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.train import NovelTrainer, TrainConfig


@pytest.fixture
def temp_config():
    config_data = {
        "model": {
            "name": "Qwen3-4B",
            "trust_remote_code": True,
        },
        "training": {
            "method": "qlora",
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "per_device_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "learning_rate": 1e-4,
            "num_epochs": 3,
            "max_seq_length": 2048,
            "output_dir": "models/test",
            "logging_dir": "logs/test",
        },
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        temp_path = f.name

    yield temp_path

    Path(temp_path).unlink()


def test_train_config_defaults():
    config = TrainConfig()
    assert config.model_name == "Qwen3-4B"
    assert config.method == "qlora"
    assert config.lora_rank == 16
    assert config.num_epochs == 3


def test_train_config_custom():
    config = TrainConfig(
        model_name="Qwen/Qwen3-7B",
        lora_rank=32,
        num_epochs=5,
    )
    assert config.model_name == "Qwen/Qwen3-7B"
    assert config.lora_rank == 32
    assert config.num_epochs == 5


def test_trainer_init(temp_config):
    trainer = NovelTrainer(temp_config)
    assert trainer.train_config is not None
    assert trainer.model_name == "Qwen3-4B"


def test_trainer_loads_config_params(temp_config):
    trainer = NovelTrainer(temp_config)
    assert trainer.train_config.lora_rank == 16
    assert trainer.train_config.lora_alpha == 32
    assert trainer.train_config.learning_rate == 1e-4


def test_trainer_target_modules_default():
    config = TrainConfig()
    expected_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    assert config.target_modules == expected_modules


def test_trainer_custom_target_modules():
    config = TrainConfig(target_modules=["q_proj", "v_proj"])
    assert config.target_modules == ["q_proj", "v_proj"]
