# Raspberry Pi Pico EEG Setup Guide

This guide explains how to connect a Raspberry Pi Pico to collect real EEG data for the blink-controlled game.

## Current Status

**Currently**: The system uses **simulated EEG data** based on your real datasets.

**To use real Pico**: Follow the steps below to connect actual hardware.

## Hardware Requirements

1. **Raspberry Pi Pico** (or Pico W)
2. **EEG sensor/electrodes** (e.g., AD8232 ECG/EEG sensor module)
3. **Jumper wires**
4. **USB cable** (micro USB to connect Pico to computer)
5. **Breadboard** (optional, for prototyping)

## Wiring Diagram

```
Raspberry Pi Pico    <->    AD8232 EEG Sensor
==================          ==================
3.3V (Pin 36)       <->     VCC
GND (Pin 38)        <->     GND
GPIO 26 (Pin 31)    <->     OUTPUT
GPIO 27 (Pin 32)    <->     LO+ (Lead Off +)
GPIO 28 (Pin 34)    <->     LO- (Lead Off -)
```

## Software Setup

### 1. Install MicroPython on Pico

1. Download MicroPython firmware from: https://micropython.org/download/rp2-pico/
2. Hold BOOTSEL button while connecting Pico to computer
3. Copy the `.uf2` file to the Pico drive
4. Pico will reboot and appear as a serial device

### 2. Upload EEG Reader Code to Pico

Create a file called `main.py` on the Pico with this content:

```python
import machine
import time
import json
from machine import ADC, Pin

# Initialize ADC for EEG reading (GPIO 26)
adc = ADC(Pin(26))

# Lead-off detection pins
lo_plus = Pin(27, Pin.IN)
lo_minus = Pin(28, Pin.IN)

# Status LED
led = Pin(25, Pin.OUT)

def read_eeg():
    """Read EEG value from ADC."""
    # Check lead-off detection
    if lo_plus.value() == 1 or lo_minus.value() == 1:
        return None  # Electrodes not connected properly
    
    # Read raw ADC value (0-65535)
    raw_value = adc.read_u16()
    
    # Convert to voltage (0-3.3V)
    voltage = raw_value * 3.3 / 65535
    
    # Convert to EEG-like value (adjust based on your sensor)
    # This scaling should match your training data range
    eeg_value = (voltage - 1.65) * 20000  # Center around 0, scale up
    
    return eeg_value

def main():
    print("Pico EEG Reader Started")
    sample_count = 0
    
    while True:
        try:
            # Read EEG value
            eeg_value = read_eeg()
            
            if eeg_value is not None:
                timestamp = time.ticks_ms()
                
                # Send data as JSON
                data = {
                    "timestamp": timestamp,
                    "eeg_value": eeg_value
                }
                print(json.dumps(data))
                
                # Blink LED to show activity
                if sample_count % 50 == 0:  # Every 50 samples
                    led.toggle()
                
                sample_count += 1
            else:
                # Electrodes disconnected
                print(json.dumps({"error": "electrodes_disconnected"}))
                led.on()  # Keep LED on to indicate error
            
            # Sample rate control (100 Hz)
            time.sleep_ms(10)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(json.dumps({"error": str(e)}))
            time.sleep(1)

if __name__ == "__main__":
    main()
```

### 3. Find the Pico's Serial Port

**Windows:**
- Open Device Manager
- Look for "USB Serial Device" under "Ports (COM & LPT)"
- Note the COM port (e.g., COM3, COM4)

**Linux/Mac:**
- Run: `ls /dev/tty*`
- Look for `/dev/ttyACM0` or `/dev/ttyUSB0`

## Running the Game with Real Pico Data

### Option 1: Command Line
```bash
# Navigate to the blink_game directory
cd blink_game

# Run with simulated data (default)
python src/game_with_pico.py

# Run with real Pico data
python src/game_with_pico.py --pico --port COM3
```

### Option 2: Modify main.py
Edit `src/main.py` to add Pico support:

```python
def run_game_with_pico():
    """Run the game with real Pico data."""
    import sys
    
    # Get port from user
    port = input("Enter Pico serial port (e.g., COM3): ").strip()
    if not port:
        port = "COM3"  # Default
    
    print(f"Starting game with Pico data from {port}")
    
    from game_with_pico import GameWithPico
    game = GameWithPico(use_pico=True, pico_port=port)
    game.run()

# Add this option to the main menu
```

## Testing the Connection

### 1. Test Pico Connection
```bash
python src/pico_eeg_reader.py
```

This will:
- Try to connect to the Pico
- Read data for 10 seconds
- Display EEG values in real-time

### 2. Expected Output
```
Testing Pico EEG Reader...
Connected to Pico on COM3 at 115200 baud
Reading data for 10 seconds...
EEG: 42350.23 at 1234567ms
EEG: 41890.45 at 1234577ms
EEG: 38920.12 at 1234587ms
...
```

## Troubleshooting

### Common Issues

1. **"Failed to connect to Pico"**
   - Check USB cable connection
   - Verify correct COM port
   - Make sure Pico is running MicroPython

2. **"No data received"**
   - Check wiring connections
   - Verify EEG sensor is powered
   - Check electrode placement

3. **"Electrodes disconnected"**
   - Ensure electrodes are properly attached to skin
   - Check electrode gel/conductivity
   - Verify LO+/LO- pin connections

4. **Poor blink detection**
   - Adjust electrode placement (forehead/temple area)
   - Check signal quality in EEG visualization
   - May need to retrain model with your specific hardware

### Signal Quality Tips

1. **Electrode Placement:**
   - Place one electrode on forehead (above eyebrow)
   - Place second electrode on temple
   - Ground electrode on earlobe or mastoid

2. **Improve Signal:**
   - Clean skin with alcohol before applying electrodes
   - Use conductive gel if available
   - Minimize movement during recording
   - Avoid electrical interference (phones, WiFi)

## Model Retraining for Your Hardware

If the pre-trained model doesn't work well with your Pico setup:

1. **Collect new data:**
   ```bash
   python src/pico_eeg_reader.py > my_eeg_data.csv
   ```

2. **Label the data** (mark blink vs normal periods)

3. **Retrain the model:**
   ```bash
   python src/model_training.py
   ```

## Next Steps

1. Set up the hardware according to the wiring diagram
2. Upload the MicroPython code to your Pico
3. Test the connection using the test script
4. Run the game with `--pico` flag
5. Enjoy playing with real EEG blink control!

## Safety Notes

- Use only low-voltage, battery-powered circuits near the head
- Ensure proper electrical isolation
- Do not use with medical-grade equipment without proper certification
- This is for educational/entertainment purposes only
