"""
推理生成模块

提供小说续写功能，支持：
- 命令行交互式生成
- 单次文本生成
- 可配置的生成参数（temperature、top_p、top_k 等）
"""

import os
import sys
from pathlib import Path
from typing import Optional, List
import argparse

import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel


class NovelGenerator:
    """
    小说生成器

    负责加载微调后的模型并生成小说续写内容
    """

    def __init__(
        self,
        model_path: str = "models/novel-qlora",
        config_path: str = "config/config.yaml",
    ):
        """
        初始化生成器

        Args:
            model_path: 模型路径
            config_path: 配置文件路径
        """
        self.model_path = model_path
        self.config_path = config_path

        # 加载配置
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        inference_config = config.get("inference", {})
        self.max_new_tokens = inference_config.get("max_new_tokens", 2048)
        self.temperature = inference_config.get("temperature", 0.7)
        self.top_p = inference_config.get("top_p", 0.8)
        self.top_k = inference_config.get("top_k", 20)
        self.repetition_penalty = inference_config.get("repetition_penalty", 1.1)

        self.model = None
        self.tokenizer = None

    def load_model(self):
        """
        加载预训练模型和分词器

        如果模型不存在会抛出 FileNotFoundError
        """
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        print(f"Loading model from {self.model_path}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )

        # 设置 pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        self.model.eval()
        print("Model loaded successfully!")

    def generate(self, prompt: str, style_prompt: str = "") -> str:
        """
        生成小说续写

        Args:
            prompt: 续写的开头文本
            style_prompt: 风格描述（可选）

        Returns:
            生成的续写文本
        """
        # 延迟加载模型
        if self.model is None:
            self.load_model()

        # 构建完整的 prompt
        if style_prompt:
            full_prompt = f"请根据以下风格续写小说：{style_prompt}\n\n请续写：{prompt}"
        else:
            full_prompt = f"请续写以下内容：\n{prompt}\n\n续写："

        # tokenize 输入
        inputs = self.tokenizer(full_prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # 配置生成参数
        generation_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # 生成
        with torch.no_grad():
            outputs = self.model.generate(**inputs, generation_config=generation_config)

        # 解码输出，去除输入部分
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return response

    def chat(self, system_prompt: str = "你是一个小说作家助手"):
        """
        交互式聊天模式

        用户可以输入文本获取续写，也可以设置写作风格
        """
        print("=" * 50)
        print("小说续写助手")
        print("输入 'quit' 或 'exit' 退出")
        print("输入 'set style <描述>' 设置写作风格")
        print("=" * 50)

        style_prompt = ""

        while True:
            try:
                user_input = input("\n请输入小说开头或需求: ").strip()

                if user_input.lower() in ["quit", "exit", "q"]:
                    print("再见!")
                    break

                # 设置风格
                if user_input.lower().startswith("set style "):
                    style_prompt = user_input[10:].strip()
                    print(f"写作风格已设置为: {style_prompt}")
                    continue

                if not user_input:
                    continue

                print("\n生成中...")
                result = self.generate(user_input, style_prompt)
                print("\n" + "=" * 50)
                print("续写结果:")
                print("=" * 50)
                print(result)

            except KeyboardInterrupt:
                print("\n\n再见!")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(description="Novel generation inference")
    parser.add_argument(
        "--model", type=str, default="models/novel-qlora", help="Model path"
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Config file"
    )
    parser.add_argument("--prompt", type=str, help="Prompt for generation")
    parser.add_argument("--style", type=str, default="", help="Style description")
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Interactive mode"
    )
    parser.add_argument("--max-tokens", type=int, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, help="Temperature for generation")

    args = parser.parse_args()

    generator = NovelGenerator(args.model, args.config)

    # 命令行参数覆盖配置文件
    if args.max_tokens:
        generator.max_new_tokens = args.max_tokens
    if args.temperature:
        generator.temperature = args.temperature

    # 根据参数决定运行模式
    if args.interactive or not args.prompt:
        generator.chat()
    else:
        result = generator.generate(args.prompt, args.style)
        print(result)


if __name__ == "__main__":
    main()
