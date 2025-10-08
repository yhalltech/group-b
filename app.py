# -----------------------------
# fastapi_crop_predict_single_env.py
# -----------------------------

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import torch
import torch.nn as nn
from typing import List

# -----------------------------
# 1️⃣ Define MLP
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
            nn.Sigmoid()  # multi-label
        )

    def forward(self, x):
        return self.model(x)

# -----------------------------
# 2️⃣ Load model and mlb
# -----------------------------
mlb = joblib.load('mlb.pkl')
model = CropMLP(input_dim=7, output_dim=len(mlb.classes_))
model.load_state_dict(torch.load('crop.pth', map_location=torch.device('cpu')))
model.eval()

# -----------------------------
# 3️⃣ FastAPI setup
# -----------------------------
app = FastAPI(title="Crop Prediction API 🌱")

class EnvFeatures(BaseModel):
    features: List[float]  # Single environment
    top_n: int = 6         # Default top 6 crops

# -----------------------------
# 4️⃣ Prediction function
# -----------------------------
def predict_crops(env_features: List[float], top_n: int = 6):
    if len(env_features) != 7:
        return {"error": "Environment must have 7 features.", "env": env_features}
    
    env_tensor = torch.tensor(env_features, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        probs = model(env_tensor).numpy().flatten()
    top_indices = probs.argsort()[-top_n:][::-1]
    return {"env": env_features, "top_crops": [mlb.classes_[i] for i in top_indices]}

# -----------------------------
# 5️⃣ API route
# -----------------------------
@app.post("/predict")
def predict(env: EnvFeatures):
    return predict_crops(env.features, top_n=env.top_n)

# -----------------------------
# 6️⃣ Optional root route
# -----------------------------
@app.get("/")
def root():
    return {"message": "Welcome to Crop Prediction API 🌱. Use POST /predict with features and top_n."}
