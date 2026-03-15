import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from diffusers import (
    StableDiffusionPipeline,
    DDPMScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
)
from diffusers.optimization import get_cosine_schedule_with_warmup
from transformers import CLIPTextModel, CLIPTokenizer
import torch.multiprocessing as mp


# ========= CONFIG =========
DATA_DIR = os.path.join("..", "data", "images")
CAPTION_FILE = os.path.join("..", "data", "captions.txt")
OUTPUT_DIR = os.path.join("..", "models", "laion-mini")
MODEL_ID = "runwayml/stable-diffusion-v1-5"
EPOCHS = 4
BATCH_SIZE = 1  # ✅ safer for RTX 3060 (12 GB)
LEARNING_RATE = 5e-6
IMAGE_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0  # ✅ change to 2-4 after testing (Windows-safe)

# ==========================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"📁 Using data from: {DATA_DIR}")
    print(f"💾 Model outputs will be saved to: {OUTPUT_DIR}")
    print(f"🚀 Training on: {DEVICE}")

    # ---------- Dataset Class ----------
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    class LAIONDataset(torch.utils.data.Dataset):
        def __init__(self, img_dir, caption_file, tokenizer, transform):
            self.img_dir = img_dir
            with open(caption_file, "r", encoding="utf-8") as f:
                self.samples = [line.strip().split("|", 1) for line in f if "|" in line]
            self.tokenizer = tokenizer
            self.transform = transform

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            filename, caption = self.samples[idx]
            img_path = os.path.join(self.img_dir, filename)
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
            tokens = self.tokenizer(
                caption,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            )
            return {"pixel_values": image, "input_ids": tokens.input_ids.squeeze(0)}

    # ---------- Load Pretrained Components ----------
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet")

    text_encoder.to(DEVICE)
    vae.to(DEVICE)
    unet.to(DEVICE)

    # Enable memory optimization
    unet.enable_xformers_memory_efficient_attention()
    unet.enable_gradient_checkpointing()

    # ---------- Create Dataloader ----------
    dataset = LAIONDataset(DATA_DIR, CAPTION_FILE, tokenizer, transform)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True
    )

    # ---------- Optimizer and Scheduler ----------
    optimizer = torch.optim.AdamW(unet.parameters(), lr=LEARNING_RATE)
    num_training_steps = len(dataloader) * EPOCHS
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, 0, num_training_steps)

    # ---------- Training Loop ----------
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    print("\n🚀 Starting training...")
    scaler = torch.cuda.amp.GradScaler()  # ✅ mixed precision

    for epoch in range(EPOCHS):
        unet.train()
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")

        for batch in progress:
            pixel_values = batch["pixel_values"].to(DEVICE, dtype=torch.float32)
            input_ids = batch["input_ids"].to(DEVICE)

            with torch.no_grad():
                encoder_outputs = vae.encode(pixel_values)
            latents = encoder_outputs.latent_dist.sample() * 0.18215

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.num_train_timesteps, (latents.shape[0],), device=DEVICE
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            encoder_hidden_states = text_encoder(input_ids).last_hidden_state

            with torch.cuda.amp.autocast():
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = torch.nn.functional.mse_loss(model_pred, noise)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            progress.set_postfix({"loss": float(loss.item())})

        unet.save_pretrained(os.path.join(OUTPUT_DIR, f"epoch_{epoch+1}"))
        print(f"💾 Saved checkpoint for epoch {epoch+1}")

    print("✅ Training complete!")

    # ---------- Save the full pipeline ----------
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    )
    pipe.save_pretrained(OUTPUT_DIR)
    print(f"🎉 Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    mp.freeze_support()  # ✅ Windows multiprocessing fix
    main()
