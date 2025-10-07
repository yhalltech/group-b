# -----------------------------
# async_fastapi_crop_api.py
# -----------------------------
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import joblib
import numpy as np
import asyncio

# -----------------------------
# 1️⃣ Define MLP model
# -----------------------------
class CropMLP(nn.Module):
    def __init__(self, input_dim=7, output_dim=2698):
        super(CropMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# -----------------------------
# 2️⃣ Load saved model and MultiLabelBinarizer
# -----------------------------
mlb = joblib.load("mlb.pkl")
model = CropMLP(input_dim=7, output_dim=len(mlb.classes_))
model.load_state_dict(torch.load("crop.pth", map_location=torch.device("cpu")))
model.eval()

# -----------------------------
# 3️⃣ FastAPI app
# -----------------------------
app = FastAPI(title="Async Crop Prediction API")

# Input schema
class Environment(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float
    top_n: int = 5  # optional, default top 5 crops

# -----------------------------
# 4️⃣ Asynchronous prediction function
# -----------------------------
async def async_predict_crops(env_features, top_n=5):
    # Simulate asynchronous behavior for heavy computation
    await asyncio.sleep(0)  # yield control to event loop
    env_tensor = torch.tensor(env_features, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        probs = model(env_tensor).numpy().flatten()
    top_indices = probs.argsort()[-top_n:][::-1]
    return [mlb.classes_[i] for i in top_indices]

# -----------------------------
# 5️⃣ Prediction endpoint
# -----------------------------
@app.post("/predict")
async def predict(env: Environment):
    features = [
        env.N, env.P, env.K,
        env.temperature, env.humidity,
        env.ph, env.rainfall
    ]
    crops = await async_predict_crops(features, top_n=env.top_n)
    return {"environment": features, "predicted_crops": crops}
