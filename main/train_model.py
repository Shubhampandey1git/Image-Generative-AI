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
from peft import LoraConfig, get_peft_model
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ========= CONFIG =========
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LATENT_DIR = os.path.join(BASE_DIR, "data", "latents")
CAPTION_FILE = os.path.join(BASE_DIR, "data", "captions.txt")
OUTPUT_DIR = os.path.join(BASE_DIR, "models", "laion-mini")
MODEL_ID = "runwayml/stable-diffusion-v1-5"
EPOCHS = 2
BATCH_SIZE = 1  # ✅ safer for RTX 3060 (6 GB)
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = os.cpu_count() // 2  # ✅ change to 2-4 after testing (Windows-safe)

# ==========================

# ---------- Dataset Class ----------
class LAIONDataset(torch.utils.data.Dataset):

    def __init__(self, latent_dir, caption_file, tokenizer):

        self.latent_dir = latent_dir

        with open(caption_file, "r", encoding="utf-8") as f:
            self.samples = [
                tuple(line.strip().split("|",1))
                for line in f if "|" in line
            ]

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        filename, caption = self.samples[idx]

        latent_path = os.path.join(
            self.latent_dir,
            filename + ".pt"
        )

        latents = torch.load(latent_path, map_location="cpu").float()

        tokens = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )

        return {
            "latents": latents.squeeze(0),
            "input_ids": tokens.input_ids.squeeze(0)
        }

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"📁 Using data from: {LATENT_DIR}")
    print(f"💾 Model outputs will be saved to: {OUTPUT_DIR}")
    print(f"🚀 Training on: {DEVICE}")

    # ---------- Load Pretrained Components ----------
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet")
    
    # ---------- Dataset Class Call ----------
    dataset = LAIONDataset(LATENT_DIR, CAPTION_FILE, tokenizer)
    
    # Freezing components we dont want to train
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    text_encoder.to(DEVICE, dtype=torch.float16)
    vae.to("cpu")
    unet.to(DEVICE, dtype=torch.float16)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.1,
        bias="none",
    )

    unet = get_peft_model(unet, lora_config)

    unet.print_trainable_parameters()
        
    # Enable memory optimization
    unet.enable_xformers_memory_efficient_attention()
    unet.enable_gradient_checkpointing()
    unet.set_attention_slice("max")

    # ---------- Create Dataloader ----------
    dataset = LAIONDataset(LATENT_DIR, CAPTION_FILE, tokenizer)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=False
    )

    # ---------- Optimizer and Scheduler ----------
    # ⚠️ If this crashes on Windows (it sometimes does), switch to:
    # optimizer = torch.optim.AdamW(unet.parameters(), lr=LEARNING_RATE)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, unet.parameters()),
        lr=LEARNING_RATE
    )
    num_training_steps = len(dataloader) * EPOCHS
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, 0, num_training_steps)
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    # ---------- Training Loop ----------
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    print("\n🚀 Starting training...")

    for epoch in range(EPOCHS):
        unet.train()
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")

        for batch in progress:
            latents = batch["latents"].to(
                DEVICE,
                dtype=torch.float16
            )
            input_ids = batch["input_ids"].to(DEVICE)

            noise = torch.randn_like(latents, dtype=torch.float16)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=DEVICE
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with torch.no_grad():
                encoder_hidden_states = text_encoder(input_ids).last_hidden_state

            with torch.cuda.amp.autocast():
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = torch.nn.functional.mse_loss(model_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            progress.set_postfix({"loss": float(loss.item())})

        unet.save_pretrained(os.path.join(OUTPUT_DIR, f"epoch_{epoch+1}_lora"))
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
    mp.freeze_support()  # Windows multiprocessing fix
    main()
