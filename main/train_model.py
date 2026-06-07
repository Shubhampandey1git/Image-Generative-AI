# app.py

import os
import time
import torch
import gradio as gr

from diffusers import StableDiffusionPipeline
from peft import PeftModel

# ================= CONFIG =================

BASE_MODEL = "runwayml/stable-diffusion-v1-5"

LORA_PATH = r"E:\AI ML\AI Image Generation\models\laion-mini\epoch_2_lora"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================

print("🚀 Loading Stable Diffusion...")

pipe = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    safety_checker=None,
    requires_safety_checker=False,
    load_safety_checker=False
)

print("🔗 Loading LoRA weights...")

pipe.unet = PeftModel.from_pretrained(
    pipe.unet,
    LORA_PATH
)

pipe = pipe.to(DEVICE)

# ---------- Optimizations ----------

pipe.enable_attention_slicing()
pipe.enable_vae_slicing()

if DEVICE == "cuda":
    pipe.enable_xformers_memory_efficient_attention()

print("✅ Model Ready!")

# ==========================================
# IMAGE GENERATION
# ==========================================

def generate_image(
    prompt,
    negative_prompt,
    steps,
    guidance,
    size,
    seed
):

    if seed == -1:
        seed = torch.randint(0, 999999, (1,)).item()

    generator = torch.Generator(device=DEVICE).manual_seed(int(seed))

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        height=int(size),
        width=int(size),
        generator=generator
    ).images[0]

    os.makedirs("outputs", exist_ok=True)

    filename = f"outputs/{int(time.time())}.png"

    image.save(filename)

    return image, filename


# ==========================================
# GRADIO UI
# ==========================================

with gr.Blocks() as demo:

    gr.Markdown("# 🎨 AI Image Generator")
    gr.Markdown("Stable Diffusion + Your Custom LoRA")

    with gr.Row():

        with gr.Column():

            prompt = gr.Textbox(
                label="Prompt",
                placeholder="A futuristic cyberpunk city at night"
            )

            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                value="blurry, low quality, distorted"
            )

            steps = gr.Slider(
                minimum=10,
                maximum=50,
                value=25,
                step=1,
                label="Inference Steps"
            )

            guidance = gr.Slider(
                minimum=1,
                maximum=15,
                value=7.5,
                step=0.5,
                label="Guidance Scale"
            )

            size = gr.Slider(
                minimum=256,
                maximum=768,
                value=512,
                step=64,
                label="Image Size"
            )

            seed = gr.Number(
                value=-1,
                label="Seed (-1 = Random)"
            )

            generate_btn = gr.Button("Generate Image")

        with gr.Column():

            output_image = gr.Image(label="Generated Image")

            output_file = gr.Textbox(label="Saved File")

    generate_btn.click(
        fn=generate_image,
        inputs=[
            prompt,
            negative_prompt,
            steps,
            guidance,
            size,
            seed
        ],
        outputs=[
            output_image,
            output_file
        ]
    )

# ==========================================

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )