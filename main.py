from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch
from torchvision import models, transforms
import time
import math

app = FastAPI()

# Load ResNet18 model once at startup
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()

# Standard ImageNet preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

with open("imagenet_labels.txt") as f:
    imagenet_labels = [line.strip() for line in f.readlines()]

# CPU burner
def burn_cpu(duration=0.3):
    start = time.time()
    while time.time() - start < duration:
        math.sqrt(12345 * 67890)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Burn CPU to trigger HPA scaling
    burn_cpu()

    # Read and preprocess image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0)

    # Run inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        top_prob, top_idx = torch.max(probs, dim=1)

    breed_name = imagenet_labels[top_idx.item()]

    return {
        "breed": breed_name,
        "probability": float(top_prob.item())
    }

@app.get("/health")
async def health():
    return {"status": "ok"}
