import os
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import argparse

import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm


@dataclass
class DataConfig:
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    min_text_length: int = 100
    max_text_length: int = 10000
    chunk_size: int = 512
    overlap: int = 50


class NovelDatasetProcessor:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.data_config = DataConfig()

    def load_raw_texts(self, raw_dir: Optional[str] = None) -> List[str]:
        raw_dir = raw_dir or self.data_config.raw_dir
        texts = []

        for file in Path(raw_dir).glob("**/*.txt"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    content = f.read()
                    if len(content) >= self.data_config.min_text_length:
                        texts.append(content)
            except Exception as e:
                print(f"Error reading {file}: {e}")

        for file in Path(raw_dir).glob("**/*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and "text" in item:
                                text = item["text"]
                            elif isinstance(item, str):
                                text = item
                            else:
                                continue
                            if len(text) >= self.data_config.min_text_length:
                                texts.append(text)
                    elif isinstance(data, dict):
                        if "texts" in data:
                            for text in data["texts"]:
                                if len(text) >= self.data_config.min_text_length:
                                    texts.append(text)
            except Exception as e:
                print(f"Error reading {file}: {e}")

        print(f"Loaded {len(texts)} texts from {raw_dir}")
        return texts

    def create_training_data(
        self, texts: List[str], prompt_template: str
    ) -> List[Dict[str, str]]:
        training_data = []

        for text in tqdm(texts, desc="Creating training data"):
            chunks = self._split_into_chunks(text)

            for i in range(len(chunks) - 1):
                prompt = chunks[i]
                continuation = chunks[i + 1]

                if len(prompt) < 50 or len(continuation) < 50:
                    continue

                prompt = prompt[:200]

                formatted = prompt_template.format(prompt=prompt)

                training_data.append(
                    {
                        "prompt": formatted,
                        "response": continuation,
                        "text": formatted + continuation,
                    }
                )

        return training_data

    def _split_into_chunks(self, text: str) -> List[str]:
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.data_config.chunk_size, text_len)

            for punct in ["。", "！", "？", "\n"]:
                last_punct = text.rfind(punct, start, end)
                if last_punct > start:
                    end = last_punct + 1
                    break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - self.data_config.overlap if end < text_len else end

        return chunks

    def split_data(
        self,
        data: List[Any],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ) -> Dict[str, List]:
        random.seed(42)

        train_val, test = train_test_split(data, test_size=test_ratio, random_state=42)

        val_size = val_ratio / (train_ratio + val_ratio)
        train, val = train_test_split(train_val, test_size=val_size, random_state=42)

        return {"train": train, "val": val, "test": test}

    def save_data(self, data: Dict[str, List[Any]], output_dir: str):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for split_name, split_data in data.items():
            file_path = output_path / f"{split_name}.jsonl"
            with open(file_path, "w", encoding="utf-8") as f:
                for item in split_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            print(f"Saved {len(split_data)} samples to {file_path}")

    def process(self, raw_dir: str = "data/raw", output_dir: str = "data/processed"):
        print("Loading raw texts...")
        texts = self.load_raw_texts(raw_dir)

        if not texts:
            print("No texts found. Please add your novel data to data/raw/")
            return

        prompt_template = self.config.get("data", {}).get(
            "prompt_template", "请根据以下风格续写小说：{prompt}\n\n请续写："
        )

        print("Creating training data...")
        training_data = self.create_training_data(texts, prompt_template)

        print("Splitting data...")
        split_data = self.split_data(training_data)

        print("Saving processed data...")
        self.save_data(split_data, output_dir)

        print("Data processing complete!")
        print(f"  Train: {len(split_data['train'])} samples")
        print(f"  Val: {len(split_data['val'])} samples")
        print(f"  Test: {len(split_data['test'])} samples")


def main():
    parser = argparse.ArgumentParser(description="Process novel data for training")
    parser.add_argument(
        "--raw-dir", type=str, default="data/raw", help="Raw data directory"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/processed", help="Output directory"
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Config file"
    )
    args = parser.parse_args()

    processor = NovelDatasetProcessor(args.config)
    processor.process(args.raw_dir, args.output_dir)


if __name__ == "__main__":
    main()
