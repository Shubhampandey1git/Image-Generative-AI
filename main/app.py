# app.py
import gradio as gr
from diffusers import StableDiffusionPipeline
import torch

# Load your Stable Diffusion model
# You can later replace this with your fine-tuned LoRA model path
model_id = "runwayml/stable-diffusion-v1-5"

print("Loading model...")
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
).to("cuda")

# Define generation function
def generate(prompt, steps, scale):
    image = pipe(
        prompt,
        num_inference_steps=int(steps),
        guidance_scale=float(scale)
    ).images[0]
    return image

# Create Gradio interface
interface = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Enter your prompt:", placeholder="A futuristic city under pink skies"),
        gr.Slider(10, 50, value=25, step=1, label="Inference Steps"),
        gr.Slider(1, 15, value=7.5, step=0.5, label="Guidance Scale")
    ],
    outputs="image",
    title="🎨 Image Generative AI",
    description="Generate images using Stable Diffusion on your RTX 3060 GPU."
)

if __name__ == "__main__":
    interface.launch()
