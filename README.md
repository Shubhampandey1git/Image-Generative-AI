# 🎨 AI Image Generation with Stable Diffusion + LoRA

A custom AI image generation system built using Stable Diffusion v1.5, LoRA fine-tuning, Diffusers, PyTorch, FastAPI, Gradio, and Android integration.

This project demonstrates the complete workflow of training and deploying a text-to-image generative AI model:

- Dataset Preparation
- Latent Caching
- LoRA Fine-Tuning
- Model Inference
- REST API Deployment
- Gradio Web Application
- Android Client Integration

---

## 🚀 Features

### Training Pipeline

✅ Stable Diffusion v1.5 Fine-Tuning

✅ LoRA (Low-Rank Adaptation)

✅ Mixed Precision Training (FP16)

✅ Gradient Checkpointing

✅ XFormers Memory Optimization

✅ Latent Caching for Faster Training

✅ Cosine Learning Rate Scheduling

---

### Inference Pipeline

✅ Text-to-Image Generation

✅ Adjustable Guidance Scale

✅ Adjustable Inference Steps

✅ Custom Negative Prompts

✅ Seed Control

✅ Image Size Selection

✅ Generated Image Saving

---

### Deployment

✅ FastAPI Backend

✅ Gradio Web Interface

✅ Android Client Application

---

## 🛠 Tech Stack

### AI / Machine Learning

- PyTorch
- Diffusers
- Transformers
- PEFT (LoRA)
- XFormers

### Backend

- FastAPI
- Uvicorn

### Frontend

- Gradio
- Android (Jetpack Compose)

### Data Processing

- PIL
- NumPy

---

## 📂 Project Structure

```text
AI Image Generation
│
├── data/
│   ├── captions.txt
│   └── latents/
│
├── models/
│   └── laion-mini/
│       ├── epoch_1_lora/
│       └── epoch_2_lora/
│
├── main/
│   ├── build_subset.py
│   ├── cache_latents.py
│   ├── train_model.py
│   ├── api.py
│   └── app.py
│
├── outputs/
│
├── requirements.txt
│
└── README.md
```

---

## ⚙️ Training Workflow

### 1. Dataset Preparation

Build a training subset:

```bash
python main/build_subset.py
```

---

### 2. Cache Latents

Convert images into Stable Diffusion latent representations:

```bash
python main/cache_latents.py
```

This significantly reduces training time by avoiding repeated VAE encoding.

---

### 3. Train LoRA

```bash
python main/train_model.py
```

Training uses:

- Stable Diffusion v1.5
- LoRA Fine-Tuning
- FP16 Mixed Precision
- Gradient Checkpointing
- Attention Slicing

---

## 🖼 Running the Gradio App

```bash
python main/app.py
```

Open:

```text
http://localhost:7860
```

---

## 🌐 Running the API Server

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

API Documentation:

```text
http://localhost:8000/docs
```

---

## 📱 Android Integration

The project includes an Android application built with:

- Kotlin
- Retrofit
- Jetpack Compose

The Android app communicates with the FastAPI backend to generate images remotely.

---

## 📊 Training Configuration

Example configuration:

```python
EPOCHS = 4
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
MODEL_ID = "runwayml/stable-diffusion-v1-5"
```

---

## 🧠 Model Architecture

Base Model:

- Stable Diffusion v1.5

Fine-Tuning Method:

- LoRA (Low-Rank Adaptation)

Trainable Parameters:

```text
1.59 Million
```

Total Parameters:

```text
861 Million+
```

Trainable Percentage:

```text
0.185%
```

---

## 💻 Hardware Used

Training Environment:

- NVIDIA RTX 3060 (6GB)
- CUDA
- Windows
- Python 3.10

---

## 🔮 Future Improvements

- Image-to-Image Generation
- Inpainting
- ControlNet Integration
- SDXL Support
- DreamBooth Training
- Cloud Deployment
- User Authentication
- Gallery History
- Model Versioning

---

## 📸 Example Prompt

```text
A futuristic cyberpunk city at sunset,
highly detailed, cinematic lighting,
ultra realistic
```

---

## 📜 License

MIT License

---

## 👨‍💻 Author

Shubham Pandey

AI / ML Engineer

Built as a portfolio project to explore:

- Generative AI
- Diffusion Models
- LoRA Fine-Tuning
- Model Deployment
- Mobile AI Applications