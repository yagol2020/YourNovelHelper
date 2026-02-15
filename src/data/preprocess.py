"""
小说数据预处理模块

将原始小说文本转换为训练数据格式，支持 txt 和 json 格式的输入文件，
输出可用于 LoRA 微调的 prompt-response 对。
"""

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
        从指定目录加载原始文本文件

        支持两种格式：
        - .txt 文件：直接读取文件内容
        - .json 文件：支持 list 格式或包含 texts 字段的 dict 格式

        Args:
            raw_dir: 原始数据目录路径，默认为配置中的 raw_dir

        Returns:
            文本列表
        """
        raw_dir = raw_dir or self.data_config.raw_dir
        texts = []

        # 读取 txt 文件
        for file in Path(raw_dir).glob("**/*.txt"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    content = f.read()
                    if len(content) >= self.data_config.min_text_length:
                        texts.append(content)
            except Exception as e:
                print(f"Error reading {file}: {e}")

        # 读取 json 文件
        for file in Path(raw_dir).glob("**/*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # 支持 list 格式: [{"text": "..."}, ...]
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
                    # 支持 dict 格式: {"texts": ["...", ...]}
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
        将文本转换为训练数据格式

        将每段文本分割成 chunk，相邻的 chunk 组成 prompt-response 对，
        用于训练模型的续写能力。

        Args:
            texts: 原始文本列表
            prompt_template: prompt 模板字符串

        Returns:
            训练数据列表，每项包含 prompt、response 和 text 字段
        """
        training_data = []

        for text in tqdm(texts, desc="Creating training data"):
            # 将文本分割成多个 chunk
            chunks = self._split_into_chunks(text)

            # 相邻 chunk 组成训练对：chunk[i] 作为 prompt，chunk[i+1] 作为续写
            for i in range(len(chunks) - 1):
                prompt = chunks[i]
                continuation = chunks[i + 1]

                # 过滤掉太短的 chunk
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

    def _split_into_chunks(self, text: str) -> List[str]:
        """
        将文本分割成多个 chunk

        按照标点符号（句号、感叹号、问号、换行）进行断句，
        每个 chunk 长度不超过 chunk_size，相邻 chunk 之间有 overlap 重叠。

        Args:
            text: 待分割的文本

        Returns:
            chunk 列表
        """
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.data_config.chunk_size, text_len)

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
            step = end - start - self.data_config.overlap
            start = start + max(1, step)

        return chunks

    def split_data(
        self,
        data: List[Any],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ) -> Dict[str, List]:
        """
        划分训练集、验证集和测试集

        Args:
            data: 待划分的数据列表
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例

        Returns:
            包含 train、val、test 三个列表的字典
        """
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Ratios must sum to 1.0, got {total}")

        random.seed(42)

        # 先划分出测试集
        train_val, test = train_test_split(data, test_size=test_ratio, random_state=42)

        # 再从剩余数据中划分验证集
        val_size = val_ratio / (train_ratio + val_ratio)
        train, val = train_test_split(train_val, test_size=val_size, random_state=42)

        return {"train": train, "val": val, "test": test}

    def save_data(self, data: Dict[str, List[Any]], output_dir: str):
        """
        将处理后的数据保存为 JSONL 格式

        Args:
            data: 包含 train、val、test 的数据字典
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for split_name, split_data in data.items():
            file_path = output_path / f"{split_name}.jsonl"
            with open(file_path, "w", encoding="utf-8") as f:
                for item in split_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            print(f"Saved {len(split_data)} samples to {file_path}")

    def process(self, raw_dir: str = "data/raw", output_dir: str = "data/processed"):
        """
        执行完整的数据处理流程

        1. 加载原始文本
        2. 创建训练数据
        3. 划分数据集
        4. 保存结果

        Args:
            raw_dir: 原始数据目录
            output_dir: 输出目录
        """
        print("Loading raw texts...")
        texts = self.load_raw_texts(raw_dir)

        if not texts:
            print("No texts found. Please add your novel data to data/raw/")
            return

        # 从配置中获取 prompt 模板
        prompt_template = self.config.get("data", {}).get(
            "prompt_template", "请根据以下风格续写小说：{prompt}\n\n请续写："
        )

        print("Creating training data...")
        training_data = self.create_training_data(texts, prompt_template)

        if not training_data:
            print(
                "Error: No training data created. Possible causes:\n"
                "  - Texts are too short (min chunk_size: {})\n"
                "  - Chunks shorter than 50 chars (filtered out)\n"
                "  - Please check your raw data or adjust chunk_size in config".format(
                    self.data_config.chunk_size
                )
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
    """命令行入口函数"""
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
