# -----------------------------
# fastapi_crop_predict_json_upgraded.py
# -----------------------------

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import joblib
import torch
import torch.nn as nn

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
# 2️⃣ Load model and MultiLabelBinarizer
# -----------------------------
mlb = joblib.load('mlb.pkl')
model = CropMLP(input_dim=7, output_dim=len(mlb.classes_))
model.load_state_dict(torch.load('crop.pth', map_location=torch.device('cpu')))
model.eval()

# -----------------------------
# 3️⃣ FastAPI setup
# -----------------------------
app = FastAPI(title="Crop Prediction API 🌱", version="1.1")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# 4️⃣ Pydantic model for input
# -----------------------------
class EnvFeatures(BaseModel):
    features: List[float]  # Single environment
    top_n: int = 6         # Default top 6 crops

# -----------------------------
# 5️⃣ Prediction function
# -----------------------------
def predict_crops(env_features: List[float], top_n: int = 6):
    if len(env_features) != 7:
        return {"error": "Each environment must have exactly 7 features.", "env": env_features}

    env_tensor = torch.tensor(env_features, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        probs = model(env_tensor).numpy().flatten()

    top_indices = probs.argsort()[-top_n:][::-1]
    top_crops = [mlb.classes_[i] for i in top_indices]

    return {
        "env": env_features,
        "top_n": top_n,
        "top_crops": top_crops
    }

# -----------------------------
# 6️⃣ API route
# -----------------------------
@app.post("/predict")
def predict(env: EnvFeatures):
    return predict_crops(env.features, top_n=env.top_n)

# -----------------------------
# 7️⃣ Optional health check
# -----------------------------
@app.get("/health")
def health():
    return {"status": "OK", "message": "API is running 🚀"}
