import base64
import io
import os

import gradio as gr
import httpx
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8080")
API_TOKEN = os.getenv("API_TOKEN", "")


def generate_post(prompt: str) -> tuple[str, Image.Image | None, str]:
    headers: dict[str, str] = {}
    if API_TOKEN:
        headers["Authorization"] = f"Bearer {API_TOKEN}"

    with httpx.Client(timeout=300) as client:
        response = client.post(
            f"{API_URL}/generate",
            json={"prompt": prompt},
            headers=headers,
        )
        response.raise_for_status()

    data = response.json()
    stats = f"Revisions: {data['revision_count']} · LLM calls: {data['llm_calls']}"

    image: Image.Image | None = None
    image_url = data.get("image_url")
    if image_url:
        if image_url.startswith("data:image"):
            _, b64data = image_url.split(",", 1)
            image = Image.open(io.BytesIO(base64.b64decode(b64data)))
        else:
            image_bytes = httpx.get(image_url).content
            image = Image.open(io.BytesIO(image_bytes))

    return data["text"], image, stats


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
        image_output = gr.Image(label="Generated Image", type="pil")

    stats_output = gr.Textbox(label="Stats", interactive=False, lines=1)

    generate_btn.click(
        fn=generate_post,
        inputs=[prompt_input],
        outputs=[post_output, image_output, stats_output],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
