import os

import gradio as gr
import httpx
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8080")
API_TOKEN = os.getenv("API_TOKEN", "")


def generate_post(prompt: str) -> tuple[str, str | None, str]:
    headers: dict[str, str] = {}
    if API_TOKEN:
        headers["Authorization"] = f"Bearer {API_TOKEN}"

    with httpx.Client(timeout=120) as client:
        response = client.post(
            f"{API_URL}/generate",
            json={"prompt": prompt},
            headers=headers,
        )
        response.raise_for_status()

    data = response.json()
    stats = f"Revisions: {data['revision_count']} · LLM calls: {data['llm_calls']}"
    return data["text"], data.get("image_url"), stats


with gr.Blocks(title="Post-It") as demo:
    gr.Markdown("# Post-It")
    gr.Markdown(
        "Enter a topic and the agent will draft a social media post and generate a matching image."
    )

    prompt_input = gr.Textbox(
        label="Prompt",
        placeholder="Write a post about agentic AI workflows...",
        lines=3,
    )
    generate_btn = gr.Button("Generate", variant="primary")

    with gr.Row():
        post_output = gr.Textbox(label="Generated Post", lines=10)
        image_output = gr.Image(label="Generated Image")

    stats_output = gr.Textbox(label="Stats", interactive=False, lines=1)

    generate_btn.click(
        fn=generate_post,
        inputs=[prompt_input],
        outputs=[post_output, image_output, stats_output],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
