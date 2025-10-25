import serial
import joblib
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import deque
from catboost import CatBoostClassifier

# ---------------------------
# CONFIGURATION
# ---------------------------
PORT = 'COM5'           # Change if needed
BAUD = 115200
WINDOW_SIZE = 200        # Number of points to display on graph
PLOT_REFRESH = 0.05      # seconds between updates
# ---------------------------

# Load trained model and scaler
model = joblib.load("catboost_model.pkl")
scaler = joblib.load("scaler.pkl")

# Open serial connection to Pico
ser = serial.Serial(PORT, BAUD, timeout=1)
start_time = time.time()

# Rolling buffers
time_buffer = deque(maxlen=WINDOW_SIZE)
eog_buffer = deque(maxlen=WINDOW_SIZE)
pred_buffer = deque(maxlen=WINDOW_SIZE)

# Setup matplotlib live plot
plt.ion()
fig, ax = plt.subplots(figsize=(8, 4))
line_eog, = ax.plot([], [], label='EOG Signal')
line_pred, = ax.plot([], [], label='Predicted Blink (1=blink)')
ax.set_xlabel("Elapsed Time (s)")
ax.set_ylabel("EOG Value / Prediction")
ax.set_title("Real-Time EOG + Blink Detection")
ax.legend()
plt.show()

print("🔌 Connected. Streaming data from Pico...")

while True:
    line = ser.readline().decode().strip()
    if not line:
        continue

    try:
        # Convert EOG reading
        exg_value = float(line)
        elapsed_ms = (time.time() - start_time) * 1000

        # Prepare for model
        X = np.array([[elapsed_ms, exg_value]])
        X_scaled = scaler.transform(X)
        pred = int(model.predict(X_scaled)[0])

        # Send back result to Pico
        if pred == 1:
            ser.write(b"BLINK\n")
        else:
            ser.write(b"REST\n")

        # Store for plotting
        time_buffer.append(elapsed_ms / 1000)
        eog_buffer.append(exg_value)
        pred_buffer.append(pred * max(eog_buffer) * 0.9)  # scale blink line for visibility

        # Update live plot
        line_eog.set_data(time_buffer, eog_buffer)
        line_pred.set_data(time_buffer, pred_buffer)
        ax.relim()
        ax.autoscale_view()
        plt.pause(PLOT_REFRESH)

        # Console feedback
        if pred == 1:
            print("🟢 Blink detected!")
        else:
            print("⚪ Resting...")

    except Exception as e:
        print("Error:", e)
        continue
