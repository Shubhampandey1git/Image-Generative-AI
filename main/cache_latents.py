import os
import torch
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "images")
CACHE_DIR = os.path.join(BASE_DIR, "data", "latents")

os.makedirs(CACHE_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading VAE...")
vae = AutoencoderKL.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    subfolder="vae"
).to(device)

transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)
])

images = [f for f in os.listdir(DATA_DIR) if f.endswith(".jpg")]

print(f"Caching latents for {len(images)} images...")

for img_name in tqdm(images):

    img_path = os.path.join(DATA_DIR, img_name)

    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        latent = vae.encode(image).latent_dist.sample() * 0.18215

    torch.save(
        latent.cpu(),
        os.path.join(CACHE_DIR, img_name + ".pt")
    )

print("Latent caching complete!")