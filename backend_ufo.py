from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI(title="UFO Predictor API")
model = pickle.load(open("ufo-model.pkl", "rb"))

class UFOInput(BaseModel):
    seconds: float
    latitude: float
    longitude: float

@app.post("/predict")
def predict_ufo(data: UFOInput):
    prediction = model.predict([[data.seconds, data.latitude, data.longitude]])
    countries = ["Australia", "Canada", "Great Britain", "UK", "US"]
    return {"country": countries[prediction[0]]}