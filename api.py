import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# 1. Initialize the FastAPI app
app = FastAPI(title="EMG Prediction API")

# 2. Load the trained model and scaler
# These are loaded only once when the API starts
try:
    scaler = joblib.load('notebook/data/scaler.pkl')
    model = joblib.load('notebook/data/catboost_model.pkl')
    print("Model and scaler loaded successfully.")
except FileNotFoundError:
    print("Error: Could not find model or scaler files in the 'models/' directory.")
    exit()

# 3. Define the input data structure using Pydantic
# This ensures that the data sent to the API has the correct format.
# The feature names must EXACTLY match the columns your model was trained on.
class EMGFeatures(BaseModel):
    ch1_voltage_roll_mean: float
    ch1_voltage_roll_std: float
    ch1_voltage_roll_max: float
    ch1_voltage_roll_min: float
    ch2_voltage_roll_mean: float
    ch2_voltage_roll_std: float
    ch2_voltage_roll_max: float
    ch2_voltage_roll_min: float
    ch3_voltage_roll_mean: float
    ch3_voltage_roll_std: float
    ch3_voltage_roll_max: float
    ch3_voltage_roll_min: float

# 4. Create the prediction endpoint
@app.post("/predict")
async def predict(features: EMGFeatures):
    # Convert the incoming data into a pandas DataFrame
    features_df = pd.DataFrame([features.dict()])
    
    # Scale the features
    scaled_features = scaler.transform(features_df)
    
    # Make a prediction
    prediction = model.predict(scaled_features)[0]
    
    # Get the probability of the prediction
    probability = model.predict_proba(scaled_features)[0].max()
    
    # Return the result as a JSON object
    return {
        "prediction": int(prediction),
        "decision": "BLINK" if int(prediction) == 1 else "REST",
        "confidence": float(probability)
    }

# This allows you to run the API directly from the script
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)