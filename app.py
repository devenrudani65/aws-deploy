from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from joblib import load
from fastapi.middleware.cors import CORSMiddleware


model = load(r"app.joblib")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class prediction(BaseModel):
    advertising_budget : float


@app.get('/')
def testing():
    return {'message': 'Budget API is working'}

@app.post('/predict')
def predict(data : prediction):

    advertising_budget = np.array([[data.advertising_budget]])

    prediction = model.predict(advertising_budget)

    return {
        "advertising_budget" : data.advertising_budget,
        "predicted_sales" : float(prediction.item())
    }





