import pygame
import pandas as pd
import numpy as np
import joblib
import serial
from collections import deque
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

# --- 1. SETUP ---
print("Loading model and scaler...")
try:
    model = joblib.load('catboost_model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("Model and scaler loaded successfully.")
except FileNotFoundError:
    print("\nERROR: Could not find 'catboost_model.pkl' or 'scaler.pkl'.")
    print("Please make sure these files are in the same folder as this script.")
    exit()

# --- LIVE DATA SETUP ---
# !!! IMPORTANT: Replace 'COM6' with your Raspberry Pi Pico's actual COM port !!!
try:
    ser = serial.Serial('COM6', 9600, timeout=1)
    print("Successfully connected to Raspberry Pi Pico on COM6.")
except serial.SerialException as e:
    print(f"\nERROR: Could not connect to the serial port. Details: {e}")
    print("Please check the port name and ensure the Pico is connected.")
    exit()

# Create a buffer to hold the last 'window_size' readings
window_size = 50 
data_buffer = deque(maxlen=window_size)

# --- Pygame Initialization ---
pygame.init()
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("LIVE Raspberry Pi Pico EMG Game")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 50)
WHITE, BLACK, BALL_COLOR = (255, 255, 255), (0, 0, 0), (226, 88, 34)
BALL_RADIUS, GRAVITY = 30, 0.6
ball_y, ball_y_velocity, is_bouncing = SCREEN_HEIGHT - BALL_RADIUS, 0, False
current_decision, BLINK_LABEL = "REST", 1

# --- 2. MAIN GAME LOOP ---
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # --- LIVE PREDICTION LOGIC ---
    if ser.in_waiting > 0:
        try:
            # Read the incoming data from the Pico
            line = ser.readline().decode('utf-8').rstrip()
            
            # --- DEBUGGING LINE 1: See the raw data from the Pico ---
            # Watch this value in the shell. It should jump when you blink.
            print(f"Pico Raw Value: {line}")
            # ---------------------------------------------------------
            
            # Convert the Pico's 16-bit reading (0-65535) to a normalized value (0-1)
            sensor_value = float(line) / 65535.0
            
            # Use the single sensor reading for all 3 expected channels
            new_reading = [sensor_value, sensor_value, sensor_value] 
            data_buffer.append(new_reading)

            # Only make a prediction when the buffer is full
            if len(data_buffer) == window_size:
                live_df = pd.DataFrame(data=list(data_buffer), columns=['ch1_voltage', 'ch2_voltage', 'ch3_voltage'])
                
                # Create the same features the model was trained on
                features = {}
                for col in live_df.columns:
                    features[f'{col}_roll_mean'] = live_df[col].mean()
                    features[f'{col}_roll_std'] = live_df[col].std()
                    features[f'{col}_roll_max'] = live_df[col].max()
                    features[f'{col}_roll_min'] = live_df[col].min()
                
                features_df = pd.DataFrame([features])
                
                # --- DEBUGGING LINE 2: See the features sent to the model ---
                # Observe how these values change between resting and blinking.
                print("Calculated Features:\n", features_df.round(3))
                # ------------------------------------------------------------
                
                # Scale the features using the loaded scaler
                scaled_features = scaler.transform(features_df)
                
                # Make a prediction with the CatBoost model
                prediction = model.predict(scaled_features)[0]
                
                # Update the game's state based on the prediction
                if prediction == BLINK_LABEL:
                    current_decision = "BLINK"
                    if not is_bouncing: # Give a new kick only when it's not already bouncing
                        is_bouncing = True
                        ball_y_velocity = -15
                else:
                    current_decision = "REST"
                    is_bouncing = False

        except (ValueError, IndexError):
            # Ignore any lines that can't be converted to a number
            pass 

    # --- GAME PHYSICS & DRAWING ---
    if is_bouncing:
        ball_y_velocity += GRAVITY
        ball_y += ball_y_velocity
        if ball_y >= SCREEN_HEIGHT - BALL_RADIUS:
            ball_y = SCREEN_HEIGHT - BALL_RADIUS
            ball_y_velocity = -ball_y_velocity * 0.8 # Bounce with some energy loss
            if abs(ball_y_velocity) < 2:
                is_bouncing = False
    else:
        ball_y, ball_y_velocity = SCREEN_HEIGHT - BALL_RADIUS, 0

    screen.fill(BLACK)
    pygame.draw.circle(screen, BALL_COLOR, (SCREEN_WIDTH // 2, int(ball_y)), BALL_RADIUS)
    text_surface = font.render(f"Decision: {current_decision}", True, WHITE)
    screen.blit(text_surface, (20, 20))
    pygame.display.flip()
    clock.tick(60)

ser.close()
pygame.quit()