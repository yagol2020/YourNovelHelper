"""
小说数据预处理模块

将原始小说文本转换为训练数据格式，支持 txt 和 json 格式的输入文件，
输出可用于 LoRA 微调的 prompt-response 对。
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm


@dataclass
class DataConfig:
    """数据预处理配置"""

    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    min_text_length: int = 100
    max_text_length: int = 10000
    chunk_size: int = 512
    overlap: int = 50


def _load_text_file(args):
    """处理单个文本文件，用于多进程"""
    file_path, min_text_length = args
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        if len(content) >= min_text_length:
            return content
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return None


def _split_into_chunks_worker(args):
    """处理单个文本的chunk分割"""
    text, chunk_size, overlap = args
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        if end < text_len:
            for punct in ["。", "！", "？", "\n"]:
                last_punct = text.rfind(punct, start, end)
                if last_punct > start:
                    end = last_punct + 1
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_len:
            break
        step = end - start - overlap
        start = start + max(1, step)

    return chunks


class NovelDatasetProcessor:
    """
    小说数据集处理器

    负责加载原始文本、分割成训练数据、划分训练/验证/测试集
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """初始化处理器，加载配置文件"""
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        data_cfg = self.config.get("data", {})
        self.data_config = DataConfig(
            raw_dir=data_cfg.get("raw_dir", "data/raw"),
            processed_dir=data_cfg.get("processed_dir", "data/processed"),
            train_ratio=data_cfg.get("train_ratio", 0.8),
            val_ratio=data_cfg.get("val_ratio", 0.1),
            test_ratio=data_cfg.get("test_ratio", 0.1),
            min_text_length=data_cfg.get("min_text_length", 100),
            max_text_length=data_cfg.get("max_text_length", 10000),
            chunk_size=data_cfg.get("chunk_size", 512),
            overlap=data_cfg.get("overlap", 50),
        )

    def load_raw_texts(self, raw_dir: Optional[str] = None) -> List[str]:
        """
        从指定目录加载原始文本文件（多进程）

        支持两种格式：
        - .txt 文件：直接读取文件内容
        - .json 文件：支持 list 格式或包含 texts 字段的 dict 格式
        """
        raw_dir = raw_dir or self.data_config.raw_dir
        texts = []

        txt_files = list(Path(raw_dir).glob("**/*.txt"))

        if txt_files:
            args_list = [(str(f), self.data_config.min_text_length) for f in txt_files]

            workers = min(4, len(txt_files))
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(_load_text_file, args): args[0]
                    for args in args_list
                }
                for future in tqdm(
                    as_completed(futures), total=len(futures), desc="Loading texts"
                ):
                    result = future.result()
                    if result:
                        texts.append(result)

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
        """
        将文本转换为训练数据格式（多进程）
        """
        training_data = []

        args_list = [
            (text, self.data_config.chunk_size, self.data_config.overlap)
            for text in texts
        ]

        workers = min(4, len(texts))
        with ProcessPoolExecutor(max_workers=workers) as executor:
            all_chunks_list = list(
                tqdm(
                    executor.map(_split_into_chunks_worker, args_list),
                    total=len(texts),
                    desc="Splitting chunks",
                )
            )

        for chunks in tqdm(all_chunks_list, desc="Creating training data"):
            for i in range(len(chunks) - 1):
                prompt = chunks[i]
                continuation = chunks[i + 1]

                if len(prompt) < 50 or len(continuation) < 50:
                    continue

                prompt = prompt[:200]

                max_seq_length = self.config.get("training", {}).get(
                    "max_seq_length", 2048
                )
                template_overhead = len(prompt_template.format(prompt=""))
                max_continuation_length = (
                    max_seq_length - len(prompt) - template_overhead
                )
                if max_continuation_length <= 0:
                    continue
                continuation = continuation[:max_continuation_length]

                formatted = prompt_template.format(prompt=prompt)

                training_data.append(
                    {
                        "prompt": formatted,
                        "response": continuation,
                        "text": formatted + continuation,
                    }
                )

        return training_data

    def split_data(
        self,
        data: List[Any],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ) -> Dict[str, List]:
        """划分训练集、验证集和测试集"""
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Ratios must sum to 1.0, got {total}")

        random.seed(42)

        train_val, test = train_test_split(data, test_size=test_ratio, random_state=42)

        val_size = val_ratio / (train_ratio + val_ratio)
        train, val = train_test_split(train_val, test_size=val_size, random_state=42)

        return {"train": train, "val": val, "test": test}

    def save_data(self, data: Dict[str, List[Any]], output_dir: str):
        """将处理后的数据保存为 JSONL 格式"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for split_name, split_data in data.items():
            file_path = output_path / f"{split_name}.jsonl"
            with open(file_path, "w", encoding="utf-8") as f:
                for item in split_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            print(f"Saved {len(split_data)} samples to {file_path}")

    def process(self, raw_dir: str = "data/raw", output_dir: str = "data/processed"):
        """执行完整的数据处理流程"""
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

        if not training_data:
            print(
                f"Error: No training data created. Possible causes:\n"
                f"  - Texts are too short (min chunk_size: {self.data_config.chunk_size})\n"
                f"  - Chunks shorter than 50 chars (filtered out)\n"
                f"  - Please check your raw data or adjust chunk_size in config"
            )
            return

        print("Splitting data...")
        split_data = self.split_data(
            training_data,
            train_ratio=self.data_config.train_ratio,
            val_ratio=self.data_config.val_ratio,
            test_ratio=self.data_config.test_ratio,
        )

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
