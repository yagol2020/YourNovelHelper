import gradio as gr
import yaml
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.inference.generate import NovelGenerator


class WebUI:
    def __init__(
        self,
        model_path: str = "models/novel-qlora",
        config_path: str = "config/config.yaml",
    ):
        self.generator = NovelGenerator(model_path, config_path)
        self.loaded = False

    def load_model(self):
        if not self.loaded:
            self.generator.load_model()
            self.loaded = True

    def generate_novel(
        self, prompt, style, max_tokens, temperature, top_p, top_k, repetition_penalty
    ):
        if not prompt:
            return "请输入小说开头或提示词"

        if not self.loaded:
            self.load_model()

        try:
            result = self.generator.generate(
                prompt, style_prompt=style if style else ""
            )
            return result
        except Exception as e:
            return f"生成出错: {str(e)}"

    def launch(self, share: bool = False, port: int = 7860):
        with gr.Blocks(title="YourNovelHelper - 小说创作助手") as demo:
            gr.Markdown("# YourNovelHelper - 小说创作助手")
            gr.Markdown("基于 Qwen3-7B 微调的小说创作模型")

            with gr.Row():
                with gr.Column(scale=2):
                    prompt_input = gr.Textbox(
                        label="小说开头 / 提示词",
                        placeholder="请输入小说开头或你想要的情节...",
                        lines=5,
                    )

                    style_input = gr.Textbox(
                        label="写作风格 (可选)",
                        placeholder="例如：金庸的武侠风格、村上春树的文风...",
                        lines=2,
                    )

                    generate_btn = gr.Button("生成", variant="primary")

                with gr.Column(scale=3):
                    output = gr.Textbox(
                        label="生成结果", lines=15, show_copy_button=True
                    )

            with gr.Row():
                max_tokens = gr.Slider(
                    100, 4096, value=2048, step=100, label="最大生成长度"
                )
                temperature = gr.Slider(
                    0.1, 1.5, value=0.7, step=0.1, label="Temperature"
                )
                top_p = gr.Slider(0.1, 1.0, value=0.8, step=0.05, label="Top P")
                top_k = gr.Slider(1, 100, value=20, step=1, label="Top K")
                repetition_penalty = gr.Slider(
                    1.0, 2.0, value=1.1, step=0.05, label="Repetition Penalty"
                )

            gr.Markdown("---")
            gr.Markdown("### 使用说明")
            gr.Markdown("""
            1. 在「小说开头」中输入你想要续写的内容
            2. (可选) 在「写作风格」中描述你喜欢的风格
            3. 调整生成参数（可选）
            4. 点击「生成」按钮
            """)

            generate_btn.click(
                fn=self.generate_novel,
                inputs=[
                    prompt_input,
                    style_input,
                    max_tokens,
                    temperature,
                    top_p,
                    top_k,
                    repetition_penalty,
                ],
                outputs=output,
            )

        demo.launch(share=share, server_port=port)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Novel generation Web UI")
    parser.add_argument("--model", type=str, default="models/novel-qlora")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    args = parser.parse_args()

    webui = WebUI(args.model, args.config)
    webui.launch(share=args.share, port=args.port)


if __name__ == "__main__":
    main()
