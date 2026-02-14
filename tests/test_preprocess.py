import pytest
import json
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocess import NovelDatasetProcessor, DataConfig


@pytest.fixture
def temp_data_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_config():
    return {
        "data": {
            "raw_dir": "data/raw",
            "processed_dir": "data/processed",
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "test_ratio": 0.1,
            "min_text_length": 100,
            "max_text_length": 10000,
            "chunk_size": 512,
            "overlap": 50,
            "prompt_template": "请根据以下风格续写小说：{prompt}\n\n请续写：",
        },
        "training": {
            "max_seq_length": 2048,
        },
    }


@pytest.fixture
def processor_with_config(sample_config, temp_data_dir, monkeypatch):
    import yaml

    config_path = Path(temp_data_dir) / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(sample_config, f)

    sample_config["data"]["raw_dir"] = str(Path(temp_data_dir) / "raw")
    sample_config["data"]["processed_dir"] = str(Path(temp_data_dir) / "processed")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(sample_config, f)

    monkeypatch.chdir(temp_data_dir)
    return NovelDatasetProcessor(str(config_path))


def test_data_config_defaults():
    config = DataConfig()
    assert config.raw_dir == "data/raw"
    assert config.chunk_size == 512
    assert config.overlap == 50
    assert config.train_ratio == 0.8


def test_split_into_chunks(processor_with_config):
    text = "这是第一句话。这是第二句话！这是第三句话？这是第四句话。\n第五句话。"
    chunks = processor_with_config._split_into_chunks(text)
    assert len(chunks) > 0
    assert all(isinstance(c, str) for c in chunks)


def test_split_into_chunks_long_text(processor_with_config):
    text = "。" * 1000
    chunks = processor_with_config._split_into_chunks(text)
    assert len(chunks) > 1


def test_load_raw_texts_empty_dir(processor_with_config):
    texts = processor_with_config.load_raw_texts()
    assert texts == []


def test_load_raw_texts_txt_file(processor_with_config, temp_data_dir):
    raw_dir = Path(temp_data_dir) / "raw"
    raw_dir.mkdir()

    test_file = raw_dir / "test.txt"
    test_file.write_text("这是一段测试文本" * 50, encoding="utf-8")

    texts = processor_with_config.load_raw_texts(str(raw_dir))
    assert len(texts) == 1


def test_load_raw_texts_json_list(processor_with_config, temp_data_dir):
    raw_dir = Path(temp_data_dir) / "raw"
    raw_dir.mkdir()

    test_file = raw_dir / "test.json"
    data = [{"text": "这是第一段文本" * 50}, {"text": "这是第二段文本" * 50}]
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    texts = processor_with_config.load_raw_texts(str(raw_dir))
    assert len(texts) == 2


def test_load_raw_texts_json_dict(processor_with_config, temp_data_dir):
    raw_dir = Path(temp_data_dir) / "raw"
    raw_dir.mkdir()

    test_file = raw_dir / "test.json"
    data = {"texts": ["第一段" * 50, "第二段" * 50]}
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    texts = processor_with_config.load_raw_texts(str(raw_dir))
    assert len(texts) == 2


def test_load_raw_texts_short_text_filtered(processor_with_config, temp_data_dir):
    raw_dir = Path(temp_data_dir) / "raw"
    raw_dir.mkdir()

    short_file = raw_dir / "short.txt"
    short_file.write_text("太短", encoding="utf-8")

    long_file = raw_dir / "long.txt"
    long_file.write_text("足够长的文本" * 50, encoding="utf-8")

    texts = processor_with_config.load_raw_texts(str(raw_dir))
    assert len(texts) == 1


def test_create_training_data(processor_with_config):
    texts = ["第一句话。第二句话。第三句话。" * 100]
    training_data = processor_with_config.create_training_data(
        texts, "请续写：{prompt}"
    )
    assert len(training_data) > 0
    assert "prompt" in training_data[0]
    assert "response" in training_data[0]


def test_split_data(processor_with_config):
    data = [{"id": i} for i in range(100)]
    result = processor_with_config.split_data(
        data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
    )

    assert len(result["train"]) == 80
    assert len(result["val"]) == 10
    assert len(result["test"]) == 10


def test_split_data_ratio_validation(processor_with_config):
    data = [{"id": i} for i in range(100)]

    with pytest.raises(ValueError):
        processor_with_config.split_data(
            data, train_ratio=0.5, val_ratio=0.5, test_ratio=0.2
        )


def test_save_data(processor_with_config, temp_data_dir):
    data = {
        "train": [{"prompt": "test", "response": "result"}],
        "val": [{"prompt": "test2", "response": "result2"}],
    }
    output_dir = Path(temp_data_dir) / "output"

    processor_with_config.save_data(data, str(output_dir))

    assert (output_dir / "train.jsonl").exists()
    assert (output_dir / "val.jsonl").exists()

    with open(output_dir / "train.jsonl", "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["prompt"] == "test"


def test_process_full_flow(processor_with_config, temp_data_dir):
    raw_dir = Path(temp_data_dir) / "raw"
    raw_dir.mkdir()

    test_file = raw_dir / "novel.txt"
    text = (
        "第一章\n\n故事开始于此。主角走在街上。天气很阴沉。他遇到了一个人。这是一个很长很长的段落。"
        * 50
    )
    test_file.write_text(text, encoding="utf-8")

    output_dir = Path(temp_data_dir) / "processed"

    processor_with_config.process(str(raw_dir), str(output_dir))

    assert (output_dir / "train.jsonl").exists()
    assert (output_dir / "val.jsonl").exists()
    assert (output_dir / "test.jsonl").exists()


def test_process_empty_dir(processor_with_config, temp_data_dir):
    raw_dir = Path(temp_data_dir) / "empty_raw"
    raw_dir.mkdir()

    output_dir = Path(temp_data_dir) / "processed"

    processor_with_config.process(str(raw_dir), str(output_dir))

    assert not (output_dir / "train.jsonl").exists()
