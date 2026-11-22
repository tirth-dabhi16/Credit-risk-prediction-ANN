import pandas as pd
import torch
import torch.nn as nn
import joblib
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR/"credit_risk_model.pkl"

# ANN Architecture
class CreditRiskANN(nn.Module):
    def __init__(self,input_features):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features,32),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(32,16),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(16,1)
        )

    def forward(self,x):
        return self.network(x)

# Load saved model data
model_data = joblib.load(MODEL_PATH)
preprocessor = model_data["preprocessor"]
lower = model_data["lower"]
upper = model_data["upper"]
input_features = model_data["input_features"]

model = CreditRiskANN(input_features)
model.load_state_dict(model_data["model_state_dict"])
model.eval()

# FastAPI
app = FastAPI(
    title="Credit Risk Prediction API",
    description="API to predict status of loan",
    version="1.0.0"
)

class CreditRiskInput(BaseModel):
    person_age : int = Field(ge=20,le=100)
    person_income : float = Field(gt=0)
    person_home_ownership : str
    person_emp_length : float | None = None
    loan_intent : str
    loan_grade : str
    loan_amnt : float = Field(gt=0)
    loan_int_rate : float | None = None
    loan_percent_income : float = Field(ge=0)
    cb_person_default_on_file : str
    cb_person_cred_hist_length : int = Field(ge=0)

@app.get("/")
def home():
    return {
        "message" : "Credit Risk Prediction API is running"
    }

@app.post("/predict")
def predict(data : CreditRiskInput):
    try:
        input_df = pd.DataFrame([data.model_dump()]) # Convert input info dataframe
        input_df["person_income"] = input_df["person_income"].clip(lower,upper) # Apply saved income clipping limits
        processed_data = preprocessor.transform(input_df) # Saved preprocessing
    
        if hasattr(processed_data,"toarray"):
            processed_data = processed_data.toarray() # # Convert sparse matrix into a dense array

        input_tensor = torch.tensor(processed_data,dtype=torch.float32) # Convert into pytorch tensor

        with torch.no_grad():
            logits = model(input_tensor)
            probability = torch.sigmoid(logits).item()
            prediction = int(probability >= 0.5) # Apply threshold

        return {
            "prediction" : prediction,
            "risk" : "Default" if prediction == 1 else "No-Default",
            "default_probability" : round(probability,4)
    }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed due to internal error"
        )







