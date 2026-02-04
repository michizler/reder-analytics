import json
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pickle
from typing import List, Dict, Any
import uvicorn
import pandas as pd
import os


app = FastAPI(title="Reder API", version="1.0.0")

# define request body
class PredictionRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(
        ...,
        example=[{
            "Segment_Segment C": 0.0,
            "unique_pages": 13,
            "click_rate": 0.888889,
            "last_interaction_date_day": 25,
            "is_negative": 1,
            "emails_clicked": 28,
            "last_interaction_date_year": 2021,
            "customer_segment": 0,
            "nps_category_Promoter": 0.0,
            "last_interaction_date_month": 7,
            "nps_category_Passive": 0.0,
            "nps_category_Detractor": 1.0,
            "engagement_ratio": 0.300000,
            "engagement_intensity": 735,
            "NPS": 3,
            "TimeSpent(minutes)": 15,
            "total_interactions": 4,
            "total_late_payments": 40,
            "payment_risk_score": 400.0,
            "late_payment_rate": 10.00
        }]
    )

def load_model():
    model_path = os.path.join('model', 'model.pkl')
    features_path = os.path.join('model', 'data.json')
    
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
        
    with open(features_path, 'r') as file:
        features = json.load(file)
        
    return model, features

@app.get('/')
def home_root():
    return {"message": "Welcome to Reder Analytics API"}

@app.post('/predict')
def model_predict(req: PredictionRequest):
    # convert request to dataframe to be used by model
    df = pd.DataFrame(req.records)
    
    #load model for prediction
    model, features = load_model()
    
    # ensure all expected features are present in the input data
    df = df.reindex(columns=features, fill_value=0)
    
    # make prediction using the model
    prediction = model.predict(df)
    pred_proba = model.predict_proba(df)[:, 1]
    
    print(prediction)
    print(type(prediction))
    
    print(pred_proba)
    print(type(pred_proba))
    
    # return prediction and pred_proba as response
    return {"prediction": int(prediction[0]), "prediction probability": float(pred_proba[0])}


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=3000, reload=True)