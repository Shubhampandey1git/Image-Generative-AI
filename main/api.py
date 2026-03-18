from fastapi import FastAPI
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline
from peft import PeftModel
import torch
import base64
from io import BytesIO

app = FastAPI()

# ========= CONFIG =========
MODEL_ID = "runwayml/stable-diffusion-v1-5"
LORA_PATH = "../models/laion-mini/epoch_2_lora"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model once
print("Loading model...")
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16
).to(DEVICE)

pipe.enable_attention_slicing()
pipe.enable_vae_slicing()

pipe.unet = PeftModel.from_pretrained(pipe.unet, LORA_PATH)

print("Model ready!")

# Request format
class PromptRequest(BaseModel):
    prompt: str
    steps: int = 25
    scale: float = 7.5

@app.post("/generate")
def generate(req: PromptRequest):

    image = pipe(
        prompt=req.prompt,
        num_inference_steps=req.steps,
        guidance_scale=req.scale
    ).images[0]

    # Convert image → base64
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return {"image": img_str}