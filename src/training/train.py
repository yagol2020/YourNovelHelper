"""
模型训练模块

使用 LoRA/QLoRA 技术微调 Qwen3-4B 模型，支持：
- 从预处理后的数据加载训练集和验证集
- 配置 LoRA 参数进行高效微调
- 导出合并后的完整模型
"""

import sys
import logging
import os
from pathlib import Path
from dataclasses import dataclass, field

import yaml
import torch
from transformers import logging as transformers_logging

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS"] = "1"
os.environ["MODELSCOPE_SDK_NO_PROGRESS_BAR"] = "1"
os.environ["MODELSCOPE_DOWNLOAD_PROGRESS"] = "0"
os.environ["TQDM_DISABLE"] = "1"

transformers_logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("modelscope").setLevel(logging.ERROR)
logging.getLogger("peft").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").propagate = False
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
from transformers import BitsAndBytesConfig
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
    max_steps: int = -1
    warmup_steps: int = 100
    logging_steps: int = 5
    save_steps: int = 100
    eval_steps: int = 100
    max_seq_length: int = 2048

    # 训练选项
    bf16: bool = True
    gradient_checkpointing: bool = True
    optim: str = "adamw_torch"
    report_to: list = field(default_factory=lambda: ["none"])

    # 输出配置
    output_dir: str = "models/checkpoints"
    logging_dir: str = field(
        default_factory=lambda: os.environ.get("TENSORBOARD_LOG_DIR", "logs")
    )


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
                field_type = type(getattr(self.train_config, key))
                try:
                    if field_type == float and isinstance(value, str):
                        value = float(value)
                    setattr(self.train_config, key, value)
                except (ValueError, TypeError):
                    pass

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
        use_local = Path(self.model_name).exists()

        if use_local:
            print(f"Loading tokenizer from local: {self.model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                padding_side="right",
            )
        else:
            if not MODELSCOPE_AVAILABLE:
                raise ImportError(
                    f"Model not found at {self.model_name} and modelscope is not installed. "
                    "Please either provide a local model or install modelscope: pip install modelscope"
                )
            model_id = (
                self.model_name
                if "/" in str(self.model_name)
                else f"qwen/{self.model_name}"
            )
            print(f"Loading tokenizer from ModelScope: {model_id}...")
            tokenizer = MsAutoTokenizer.from_pretrained(
                model_id,
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
        use_local = Path(self.model_name).exists()

        import io
        import sys
        from contextlib import redirect_stdout, redirect_stderr

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        with open(os.devnull, "w") as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                if use_local:
                    print(f"Loading model from local: {self.model_name}...")
                    model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        trust_remote_code=self.trust_remote_code,
                        torch_dtype=torch.bfloat16
                        if self.train_config.bf16
                        else torch.float32,
                        device_map="auto",
                        quantization_config=quantization_config,
                    )
                else:
                    if not MODELSCOPE_AVAILABLE:
                        raise ImportError(
                            f"Model not found at {self.model_name} and modelscope is not installed. "
                            "Please either provide a local model or install modelscope: pip install modelscope"
                        )
                    model_id = (
                        self.model_name
                        if "/" in str(self.model_name)
                        else f"qwen/{self.model_name}"
                    )
                    print(f"Loading model from ModelScope: {model_id}...")
                    model = MsAutoModelForCausalLM.from_pretrained(
                        model_id,
                        trust_remote_code=self.trust_remote_code,
                        torch_dtype=torch.bfloat16
                        if self.train_config.bf16
                        else torch.float32,
                        device_map="auto",
                        quantization_config=quantization_config,
                    )

        model = self._setup_lora(model)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        print(
            f"trainable params: {trainable_params:,} || all params: {all_params:,} || trainable%: {100 * trainable_params / all_params:.4f}"
        )

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

        支持缓存：如果 tokenized 数据已存在且源文件未修改，则直接加载缓存。

        Args:
            tokenizer: 分词器对象

        Returns:
            包含 train 和 validation 数据集的字典
        """
        import hashlib
        import time

        data_config = {
            "train": "data/processed/train.jsonl",
            "validation": "data/processed/val.jsonl",
        }

        cache_dir = Path("data/processed/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)

        def get_file_hash(path: Path) -> str:
            """获取文件内容hash用于判断是否需要重新tokenize"""
            if not path.exists():
                return ""
            with open(path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()[:16]

        def tokenize_function(examples):
            prompts = examples["prompt"]
            responses = examples["response"]

            prompt_encoded = tokenizer(
                prompts,
                truncation=True,
                max_length=self.train_config.max_seq_length,
                padding="max_length",
                return_tensors=None,
            )

            full_encoded = tokenizer(
                prompts,
                responses,
                truncation=True,
                max_length=self.train_config.max_seq_length,
                padding="max_length",
                return_tensors=None,
            )

            labels = []
            for i in range(len(full_encoded["input_ids"])):
                prompt_len = sum(prompt_encoded["attention_mask"][i])
                label = full_encoded["input_ids"][i][:]
                for j in range(prompt_len):
                    if j < len(label):
                        label[j] = -100
                labels.append(label)

            return {
                "input_ids": full_encoded["input_ids"],
                "labels": labels,
                "attention_mask": full_encoded["attention_mask"],
            }

        datasets = {}
        for split, path in data_config.items():
            if not Path(path).exists():
                print(f"Warning: {path} not found, skipping {split}")
                continue

            cache_file = cache_dir / f"{split}_cached"
            source_hash = get_file_hash(Path(path))
            hash_file = cache_dir / f"{split}_hash.txt"

            use_cache = False
            if cache_file.exists() and hash_file.exists():
                cached_hash = hash_file.read_text().strip()
                if cached_hash == source_hash:
                    print(f"Loading {split} dataset from cache...")
                    ds = load_dataset("json", data_files=str(cache_file), split="train")
                    ds.set_format("torch")
                    use_cache = True

            if not use_cache:
                print(f"Loading {split} dataset from {path}...")
                ds = load_dataset(
                    "json",
                    data_files=path,
                    split="train",
                    num_proc=min(4, os.cpu_count() or 4),
                )
                ds = ds.map(
                    tokenize_function,
                    batched=True,
                    batch_size=1000,
                    remove_columns=ds.column_names,
                    desc=f"Tokenizing {split}",
                    num_proc=min(4, os.cpu_count() or 4),
                )
                ds.set_format("torch")

                print(f"Saving {split} dataset to cache...")
                ds.to_json(str(cache_file))
                hash_file.write_text(source_hash)

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
            eval_strategy="steps" if "validation" in datasets else "no",
            report_to=self.train_config.report_to,
            remove_unused_columns=False,
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            max_steps=self.train_config.max_steps,
            disable_tqdm=False,
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
