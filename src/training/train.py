"""
模型训练模块

使用 LoRA/QLoRA 技术微调 Qwen3-4B 模型，支持：
- 从预处理后的数据加载训练集和验证集
- 配置 LoRA 参数进行高效微调
- 导出合并后的完整模型
"""

import os
import sys
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import yaml
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
)
from datasets import load_dataset

try:
    from modelscope import AutoModelForCausalLM as MsAutoModelForCausalLM
    from modelscope import AutoTokenizer as MsAutoTokenizer

    MODELSCOPE_AVAILABLE = True
except ImportError:
    MODELSCOPE_AVAILABLE = False


@dataclass
class TrainConfig:
    """训练配置参数"""

    model_name: str = "Qwen3-4B"
    trust_remote_code: bool = True

    # LoRA 配置
    method: str = "qlora"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    # 训练超参数
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    num_epochs: int = 3
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    max_seq_length: int = 2048

    # 训练选项
    bf16: bool = True
    gradient_checkpointing: bool = True
    optim: str = "adamw_torch"

    # 输出配置
    output_dir: str = "models/checkpoints"
    logging_dir: str = "logs"


class NovelTrainer:
    """
    小说模型训练器

    负责加载模型、数据集、配置 LoRA、运行训练和导出模型
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化训练器，加载配置文件

        Args:
            config_path: 配置文件路径
        """
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self.train_config = TrainConfig()
        # 从配置文件中覆盖默认参数
        for key, value in config.get("training", {}).items():
            if hasattr(self.train_config, key):
                setattr(self.train_config, key, value)

        self.model_name = config.get("model", {}).get(
            "name", self.train_config.model_name
        )
        self.trust_remote_code = config.get("model", {}).get("trust_remote_code", True)

    def load_tokenizer(self):
        """
        加载分词器

        Returns:
            tokenizer 对象
        """
        is_modelscope = MODELSCOPE_AVAILABLE and (
            not Path(self.model_name).exists()
            or str(self.model_name).startswith("qwen/")
            or "/" not in str(self.model_name)
        )

        model_id = (
            self.model_name
            if "/" in str(self.model_name)
            else f"qwen/{self.model_name}"
        )

        print(f"Loading tokenizer from {model_id}...")

        if is_modelscope:
            tokenizer = MsAutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=self.trust_remote_code,
                padding_side="right",
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                padding_side="right",
            )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def load_model(self, tokenizer):
        """
        加载预训练模型并应用 LoRA

        Args:
            tokenizer: 分词器对象

        Returns:
            应用 LoRA 后的模型
        """
        is_modelscope = MODELSCOPE_AVAILABLE and (
            not Path(self.model_name).exists()
            or str(self.model_name).startswith("qwen/")
            or "/" not in str(self.model_name)
        )

        model_id = (
            self.model_name
            if "/" in str(self.model_name)
            else f"qwen/{self.model_name}"
        )

        print(f"Loading model from {model_id}...")

        if is_modelscope:
            model = MsAutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=self.trust_remote_code,
                device_map="auto",
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                torch_dtype=torch.bfloat16 if self.train_config.bf16 else torch.float32,
                device_map="auto",
            )

        model = self._setup_lora(model)
        model.print_trainable_parameters()

        return model

    def _setup_lora(self, model):
        """
        配置 LoRA 参数

        Args:
            model: 基础模型

        Returns:
            应用 LoRA 后的模型
        """
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.train_config.lora_rank,
            lora_alpha=self.train_config.lora_alpha,
            lora_dropout=self.train_config.lora_dropout,
            target_modules=self.train_config.target_modules,
            bias="none",
        )

        model = get_peft_model(model, lora_config)
        return model

    def load_dataset(self, tokenizer):
        """
        加载并预处理训练数据集

        对 prompt-response 数据进行 tokenize，构建 input_ids 和 labels。
        prompt 部分使用 -100 填充，使得训练时只计算 response 的 loss。

        Args:
            tokenizer: 分词器对象

        Returns:
            包含 train 和 validation 数据集的字典
        """
        data_config = {
            "train": "data/processed/train.jsonl",
            "validation": "data/processed/val.jsonl",
        }

        def tokenize_function(examples):
            """
            tokenize 函数

            将 prompt 和 response 分别 tokenize，然后拼接。
            labels 中 prompt 部分设为 -100，只计算 response 的 loss。
            """
            prompts = examples["prompt"]
            responses = examples["response"]

            input_ids = []
            labels = []

            for prompt, response in zip(prompts, responses):
                prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
                response_ids = tokenizer(response, add_special_tokens=False)[
                    "input_ids"
                ]

                # 拼接 prompt 和 response
                input_id = prompt_ids + response_ids
                # prompt 部分设为 -100，保留 response 部分
                label = [-100] * len(prompt_ids) + response_ids

                # 截断过长的序列
                if len(input_id) > self.train_config.max_seq_length:
                    input_id = input_id[: self.train_config.max_seq_length]
                    label = label[: self.train_config.max_seq_length]

                # padding 到固定长度
                while len(input_id) < self.train_config.max_seq_length:
                    input_id.append(tokenizer.pad_token_id)
                    label.append(-100)

                input_ids.append(input_id)
                labels.append(label)

            return {"input_ids": input_ids, "labels": labels}

        datasets = {}
        for split, path in data_config.items():
            if not Path(path).exists():
                print(f"Warning: {path} not found, skipping {split}")
                continue

            print(f"Loading {split} dataset from {path}...")
            ds = load_dataset("json", data_files=path, split="train")
            ds = ds.map(
                tokenize_function,
                batched=True,
                remove_columns=ds.column_names,
                desc=f"Tokenizing {split}",
            )
            datasets[split] = ds

        return datasets

    def train(self):
        """
        执行模型训练流程

        1. 加载分词器和模型
        2. 加载数据集
        3. 配置训练参数
        4. 开始训练并保存模型
        """
        tokenizer = self.load_tokenizer()
        model = self.load_model(tokenizer)
        datasets = self.load_dataset(tokenizer)

        if not datasets:
            raise ValueError("No datasets found. Please run data preprocessing first.")

        # 配置训练参数
        training_args = TrainingArguments(
            output_dir=self.train_config.output_dir,
            per_device_train_batch_size=self.train_config.per_device_batch_size,
            per_device_eval_batch_size=self.train_config.per_device_batch_size,
            gradient_accumulation_steps=self.train_config.gradient_accumulation_steps,
            learning_rate=self.train_config.learning_rate,
            num_train_epochs=self.train_config.num_epochs,
            warmup_steps=self.train_config.warmup_steps,
            logging_dir=self.train_config.logging_dir,
            logging_steps=self.train_config.logging_steps,
            save_steps=self.train_config.save_steps,
            eval_steps=self.train_config.eval_steps,
            bf16=self.train_config.bf16,
            gradient_checkpointing=self.train_config.gradient_checkpointing,
            optim=self.train_config.optim,
            save_total_limit=3,
            load_best_model_at_end=True,
            evaluation_strategy="steps" if "validation" in datasets else "no",
            report_to=["tensorboard"],
            remove_unused_columns=False,
        )

        # 数据整理器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

        # 创建训练器
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=datasets.get("train"),
            eval_dataset=datasets.get("validation"),
            data_collator=data_collator,
        )

        print("Starting training...")
        trainer.train()

        print("Saving final model...")
        trainer.save_model(f"{self.train_config.output_dir}/final")
        tokenizer.save_pretrained(f"{self.train_config.output_dir}/final")

        print("Training complete!")

    def merge_and_export(
        self, checkpoint_path: str, export_path: str = "models/novel-qlora"
    ):
        """
        合并 LoRA 权重并导出完整模型

        将 LoRA 权重合并到基础模型中，导出可独立使用的模型文件。

        Args:
            checkpoint_path: LoRA checkpoint 路径
            export_path: 导出路径
        """
        print(f"Loading base model and checkpoint from {checkpoint_path}...")

        base_model = AutoModelForCausalLM.from_pretrained(
            self.train_config.model_name,
            trust_remote_code=self.trust_remote_code,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )

        # 加载 LoRA 并合并
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        model = model.merge_and_unload()

        print(f"Saving merged model to {export_path}...")
        model.save_pretrained(export_path)

        tokenizer = self.load_tokenizer()
        tokenizer.save_pretrained(export_path)

        print("Export complete!")


def main():
    """命令行入口函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Train novel generation model")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--export", action="store_true", help="Export merged model")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint path for export")
    parser.add_argument(
        "--output",
        type=str,
        default="models/novel-qlora",
        help="Output path for export",
    )
    args = parser.parse_args()

    trainer = NovelTrainer(args.config)

    if args.export:
        if not args.checkpoint:
            print("Error: --checkpoint required for export")
            sys.exit(1)
        trainer.merge_and_export(args.checkpoint, args.output)
    else:
        trainer.train()


if __name__ == "__main__":
    main()
