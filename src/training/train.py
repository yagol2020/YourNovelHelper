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


@dataclass
class TrainConfig:
    model_name: str = "Qwen/Qwen3-7B"
    trust_remote_code: bool = True

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

    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    num_epochs: int = 3
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    max_seq_length: int = 2048

    bf16: bool = True
    gradient_checkpointing: bool = True
    optim: str = "adamw_torch"

    output_dir: str = "models/checkpoints"
    logging_dir: str = "logs"


class NovelTrainer:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self.train_config = TrainConfig()
        for key, value in config.get("training", {}).items():
            if hasattr(self.train_config, key):
                setattr(self.train_config, key, value)

        self.model_name = config.get("model", {}).get(
            "name", self.train_config.model_name
        )
        self.trust_remote_code = config.get("model", {}).get("trust_remote_code", True)

    def load_tokenizer(self):
        print(f"Loading tokenizer from {self.model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            padding_side="right",
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def load_model(self, tokenizer):
        print(f"Loading model from {self.train_config.model_name}...")

        model = AutoModelForCausalLM.from_pretrained(
            self.train_config.model_name,
            trust_remote_code=self.trust_remote_code,
            torch_dtype=torch.bfloat16 if self.train_config.bf16 else torch.float32,
            device_map="auto",
        )

        model = self._setup_lora(model)
        model.print_trainable_parameters()

        return model

    def _setup_lora(self, model):
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
        data_config = {
            "train": "data/processed/train.jsonl",
            "validation": "data/processed/val.jsonl",
        }

        def tokenize_function(examples):
            prompts = examples["prompt"]
            responses = examples["response"]

            input_ids = []
            labels = []

            for prompt, response in zip(prompts, responses):
                prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
                response_ids = tokenizer(response, add_special_tokens=False)[
                    "input_ids"
                ]

                input_id = prompt_ids + response_ids
                label = [-100] * len(prompt_ids) + response_ids

                if len(input_id) > self.train_config.max_seq_length:
                    input_id = input_id[: self.train_config.max_seq_length]
                    label = label[: self.train_config.max_seq_length]

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
        tokenizer = self.load_tokenizer()
        model = self.load_model(tokenizer)
        datasets = self.load_dataset(tokenizer)

        if not datasets:
            raise ValueError("No datasets found. Please run data preprocessing first.")

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

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

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
        print(f"Loading base model and checkpoint from {checkpoint_path}...")

        base_model = AutoModelForCausalLM.from_pretrained(
            self.train_config.model_name,
            trust_remote_code=self.trust_remote_code,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )

        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        model = model.merge_and_unload()

        print(f"Saving merged model to {export_path}...")
        model.save_pretrained(export_path)

        tokenizer = self.load_tokenizer()
        tokenizer.save_pretrained(export_path)

        print("Export complete!")


def main():
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
