import pygame
import pandas as pd
import numpy as np
import serial
from collections import deque
import requests # Import the requests library
import warnings

warnings.filterwarnings('ignore')

# --- 1. SETUP ---
# REMOVED: joblib.load for model and scaler. The API handles this now.

# API Endpoint URL
API_URL = "http://127.0.0.1:8000/predict"

# (The rest of the LIVE DATA SETUP and Pygame Initialization is the same...)
# ...
# --- 2. MAIN GAME LOOP ---
running = True
while running:
    # (Event handling is the same...)
    # ...
    if ser.in_waiting > 0:
        try:
            # (Data reading and buffer logic is the same...)
            # ...
            if len(data_buffer) == window_size:
                live_df = pd.DataFrame(data=list(data_buffer), columns=['ch1_voltage', 'ch2_voltage', 'ch3_voltage'])
                
                features = {}
                for col in live_df.columns:
                    features[f'{col}_roll_mean'] = live_df[col].mean()
                    features[f'{col}_roll_std'] = live_df[col].std()
                    features[f'{col}_roll_max'] = live_df[col].max()
                    features[f'{col}_roll_min'] = live_df[col].min()
                
                # --- NEW PREDICTION LOGIC ---
                try:
                    # Send the features as a JSON payload to the API
                    response = requests.post(API_URL, json=features)
                    response.raise_for_status() # Raise an exception for bad status codes
                    
                    # Get the prediction from the API's response
                    result = response.json()
                    prediction = result.get("prediction")
                    current_decision = result.get("decision", "ERROR")
                    
                    if prediction == BLINK_LABEL:
                        if not is_bouncing:
                            is_bouncing = True
                            ball_y_velocity = -15
                    else:
                        is_bouncing = False
                
                except requests.exceptions.RequestException as e:
                    print(f"Could not connect to API: {e}")
                    current_decision = "API_ERROR"

        except (ValueError, IndexError):
            pass 
    
    # (The GAME PHYSICS & DRAWING part is the same...)
    # ...