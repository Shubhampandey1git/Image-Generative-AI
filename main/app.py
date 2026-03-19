# app.py
import gradio as gr
import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel
import time
import os

# ========= CONFIG =========
MODEL_ID = "runwayml/stable-diffusion-v1-5"
LORA_PATH = LORA_PATH = r"E:\AI ML\AI Image Generation\models\laion-mini\epoch_2_lora"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("🚀 Loading base model...")
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
).to(DEVICE)

# Speed optimizations
pipe.to("cuda")
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()

print("🔗 Loading LoRA...")
pipe.unet = PeftModel.from_pretrained(pipe.unet, LORA_PATH)

print("✅ Model ready!")

# ---------- Generation Function ----------
def generate(prompt, steps, scale, negative_prompt, size, seed):
    gr.Slider(256, 768, value=512, step=64, label="Image Size")
    
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=int(steps),
        guidance_scale=float(scale),
        height=size,
        width=size
    ).images[0]
    
    os.makedirs("outputs", exist_ok=True)
    
    filename = f"outputs/{int(time.time())}.png"
    image.save(filename)
    return image

# ---------- Gradio UI ----------
interface = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="A futuristic city under pink skies"),
        gr.Slider(10, 50, value=25, step=1, label="Inference Steps"),
        gr.Slider(1, 15, value=7.5, step=0.5, label="Guidance Scale"),
        gr.Textbox(label="Negative Prompt", value="blurry, low quality"),
        gr.Slider(256, 768, value=512, step=64, label="Image Size"),
        gr.Number(label="Seed", value=42)
    ],
    outputs="image",
    title="🎨 Your Custom Stable Diffusion (LoRA)",
    description="Generate images using your trained LoRA model 🚀"
)

if __name__ == "__main__":
    interface.launch()