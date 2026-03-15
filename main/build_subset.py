import os
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import itertools

# ================= CONFIG =================
SAVE_DIR = os.path.join("..", "data", "images")
CAPTIONS_FILE = os.path.join("..", "data", "captions.txt")
FAILED_LOG = os.path.join("..", "data", "failed_urls.txt")

TARGET_IMAGES = 27000   # Choose based on your goal (≈ 1GB for 500 JPEGs @512px)
THREADS = 16
TIMEOUT = 4
MAX_RETRIES = 1
BUFFER_MULTIPLIER = 100  # How many extra samples to pull to find valid ones
# ==========================================

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CAPTIONS_FILE), exist_ok=True)

print("\n📁 Directories ready.")
print(f"🎯 Target: {TARGET_IMAGES} valid images\n")

# Use a fresher LAION subset (aesthetic variant)
dataset = load_dataset("laion/laion2B-en-aesthetic", split="train", streaming=True)
print("📦 Dataset stream opened...\n")

# ---------- Helper Functions ----------
def try_download(url):
    """Try downloading an image once."""
    try:
        r = requests.get(url, timeout=TIMEOUT)
        if r.status_code == 200:
            return r.content
    except Exception:
        pass
    return None

def process_sample(sample_id, item):
    """Download and validate a single image sample."""
    url = item.get("image_url") or item.get("URL") or item.get("url")
    caption = item.get("caption") or item.get("TEXT") or ""

    if not url or not url.startswith("http"):
        return None

    content = try_download(url)
    if not content:
        with open(FAILED_LOG, "a", encoding="utf-8") as f:
            f.write(url + "\n")
        return None

    try:
        img = Image.open(BytesIO(content)).convert("RGB")
        img = img.resize((512, 512))
        out_path = os.path.join(SAVE_DIR, f"{sample_id}.jpg")
        img.save(out_path, "JPEG", quality=90)
        return (out_path, caption)
    except Exception:
        return None
# ---------------------------------------

results = []
samples = itertools.islice(dataset, TARGET_IMAGES * BUFFER_MULTIPLIER)
print(f"⚙️ Starting download loop — may skip thousands of dead URLs...\n")

with ThreadPoolExecutor(max_workers=THREADS) as executor:
    futures = []
    with tqdm(total=TARGET_IMAGES, desc="Downloading valid images", unit="img") as pbar:
        for i, item in enumerate(samples):
            if len(results) >= TARGET_IMAGES:
                break
            futures.append(executor.submit(process_sample, i, item))

            if len(futures) >= THREADS * 4:
                for future in as_completed(futures):
                    res = future.result()
                    futures.remove(future)
                    if res:
                        results.append(res)
                        pbar.update(1)
                    if len(results) >= TARGET_IMAGES:
                        break

        # Final flush
        for future in as_completed(futures):
            res = future.result()
            if res:
                results.append(res)
                pbar.update(1)
            if len(results) >= TARGET_IMAGES:
                break

# Save captions
with open(CAPTIONS_FILE, "w", encoding="utf-8") as f:
    for path, caption in results:
        f.write(f"{os.path.basename(path)}|{caption}\n")

print(f"\n✅ Done! Saved {len(results)} valid images in {SAVE_DIR}")
print(f"📝 Captions saved to {CAPTIONS_FILE}")
if os.path.exists(FAILED_LOG):
    print(f"⚠️ Failed URLs logged to {FAILED_LOG}")
