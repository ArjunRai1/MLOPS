import pandas as pd
from fastapi import FastAPI
import pickle
from watermodel import Water
app = FastAPI(
    title="Water Potability Prediction",
    description="Predicting if water is fit for consumption or not"
)

@app.get("/")
def index():
    return "Welcome to Water Potability Prediction"

@app.post("/water-potability-pred")
def predict(Water: Water):
    sample = pd.DataFrame({
    "ph": [Water.ph],
    "Hardness": [Water.Hardness],
    "Solids": [Water.Solids],
    "Chloramines": [Water.Chloramines],
    "Sulfate": [Water.Sulfate],
    "Conductivity": [Water.Conductivity],
    "Organic_carbon": [Water.Organic_carbon],
    "Trihalomethanes": [Water.Trihalomethanes],
    "Turbidity": [Water.Turbidity],
    })

    model = pickle.load(open("../model.pkl", "rb"))
    prediction = model.predict(sample)

    if prediction == 1:
        return "Water is fit for consumption"
    else:
        return "Water is not fit for consumption"