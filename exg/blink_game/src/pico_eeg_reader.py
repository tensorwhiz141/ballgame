"""
Real EEG data reader for Raspberry Pi Pico.
This module would connect to a Pico via serial/USB to get real EEG data.
"""

import serial
import time
import threading
import queue
from collections import deque
import json

class PicoEEGReader:
    def __init__(self, port='COM3', baudrate=115200, buffer_size=1000):
        """
        Initialize connection to Raspberry Pi Pico for EEG data.
        
        Args:
            port: Serial port (e.g., 'COM3' on Windows, '/dev/ttyUSB0' on Linux)
            baudrate: Serial communication speed
            buffer_size: Maximum number of samples to keep in buffer
        """
        self.port = port
        self.baudrate = baudrate
        self.buffer_size = buffer_size
        self.is_running = False
        
        # Data buffers
        self.data_buffer = deque(maxlen=buffer_size)
        self.timestamp_buffer = deque(maxlen=buffer_size)
        
        # Queue for real-time data access
        self.data_queue = queue.Queue()
        
        # Serial connection
        self.serial_connection = None
        
    def connect(self):
        """Connect to the Pico via serial."""
        try:
            self.serial_connection = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1
            )
            print(f"Connected to Pico on {self.port} at {self.baudrate} baud")
            return True
        except Exception as e:
            print(f"Failed to connect to Pico: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from the Pico."""
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
            print("Disconnected from Pico")
    
    def parse_data_line(self, line):
        """
        Parse a line of data from the Pico.
        Expected format: "timestamp,eeg_value" or JSON format
        """
        try:
            line = line.strip()
            
            # Try JSON format first
            if line.startswith('{'):
                data = json.loads(line)
                timestamp = data.get('timestamp', time.time() * 1000)
                value = data.get('eeg_value', 0)
                return timestamp, value
            
            # Try CSV format
            elif ',' in line:
                parts = line.split(',')
                timestamp = float(parts[0])
                value = float(parts[1])
                return timestamp, value
            
            # Try single value (assume current timestamp)
            else:
                timestamp = time.time() * 1000
                value = float(line)
                return timestamp, value
                
        except Exception as e:
            print(f"Error parsing data line '{line}': {e}")
            return None, None
    
    def read_thread(self):
        """Thread function to continuously read data from Pico."""
        while self.is_running and self.serial_connection and self.serial_connection.is_open:
            try:
                # Read line from serial
                if self.serial_connection.in_waiting > 0:
                    line = self.serial_connection.readline().decode('utf-8').strip()
                    
                    if line:
                        timestamp, value = self.parse_data_line(line)
                        
                        if timestamp is not None and value is not None:
                            # Add to buffers
                            self.data_buffer.append(value)
                            self.timestamp_buffer.append(timestamp)
                            
                            # Add to queue for external access
                            try:
                                self.data_queue.put({
                                    'timestamp': timestamp,
                                    'value': value,
                                    'label': 'unknown'  # Real-time data doesn't have labels
                                }, block=False)
                            except queue.Full:
                                # Remove old data if queue is full
                                try:
                                    self.data_queue.get_nowait()
                                    self.data_queue.put({
                                        'timestamp': timestamp,
                                        'value': value,
                                        'label': 'unknown'
                                    }, block=False)
                                except queue.Empty:
                                    pass
                
                time.sleep(0.001)  # Small delay to prevent excessive CPU usage
                
            except Exception as e:
                print(f"Error reading from Pico: {e}")
                time.sleep(0.1)
    
    def start(self):
        """Start reading data from the Pico."""
        if not self.connect():
            return False
        
        self.is_running = True
        self.read_thread_obj = threading.Thread(target=self.read_thread)
        self.read_thread_obj.daemon = True
        self.read_thread_obj.start()
        
        print("Started reading EEG data from Pico")
        return True
    
    def stop(self):
        """Stop reading data from the Pico."""
        self.is_running = False
        
        if hasattr(self, 'read_thread_obj'):
            self.read_thread_obj.join(timeout=1)
        
        self.disconnect()
        print("Stopped reading EEG data from Pico")
    
    def get_latest_sample(self):
        """Get the latest EEG sample (compatible with simulator interface)."""
        try:
            return self.data_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_buffer_data(self, num_samples=None):
        """Get recent data from the buffer (compatible with simulator interface)."""
        if num_samples is None:
            num_samples = len(self.data_buffer)
        
        num_samples = min(num_samples, len(self.data_buffer))
        
        if num_samples == 0:
            return [], []
        
        timestamps = list(self.timestamp_buffer)[-num_samples:]
        values = list(self.data_buffer)[-num_samples:]
        
        return timestamps, values
    
    def send_command(self, command):
        """Send a command to the Pico."""
        if self.serial_connection and self.serial_connection.is_open:
            try:
                self.serial_connection.write(f"{command}\n".encode('utf-8'))
                return True
            except Exception as e:
                print(f"Error sending command to Pico: {e}")
                return False
        return False

# Example Pico code (MicroPython) that would run on the Raspberry Pi Pico:
"""
# This code would run on the Raspberry Pi Pico
# Save as main.py on the Pico

import machine
import time
import json
from machine import ADC, Pin

# Initialize ADC for EEG reading (assuming EEG signal on GPIO 26)
adc = ADC(Pin(26))

# LED for status indication
led = Pin(25, Pin.OUT)

def read_eeg():
    # Read raw ADC value (0-65535)
    raw_value = adc.read_u16()
    
    # Convert to voltage (0-3.3V)
    voltage = raw_value * 3.3 / 65535
    
    # Convert to EEG-like value (you may need to adjust this based on your hardware)
    eeg_value = voltage * 20000  # Scale to match your dataset range
    
    return eeg_value

def main():
    print("Pico EEG Reader Started")
    
    while True:
        try:
            # Read EEG value
            eeg_value = read_eeg()
            timestamp = time.ticks_ms()
            
            # Send data as JSON
            data = {
                "timestamp": timestamp,
                "eeg_value": eeg_value
            }
            print(json.dumps(data))
            
            # Blink LED to show activity
            led.toggle()
            
            # Sample rate control (adjust as needed)
            time.sleep_ms(10)  # 100 Hz sampling rate
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()
"""

if __name__ == "__main__":
    # Test the Pico reader
    print("Testing Pico EEG Reader...")
    print("Make sure your Pico is connected and running the EEG reader code.")
    
    # You may need to change the port based on your system
    # Windows: usually COM3, COM4, etc.
    # Linux/Mac: usually /dev/ttyUSB0, /dev/ttyACM0, etc.
    
    reader = PicoEEGReader(port='COM3')  # Change this to your Pico's port
    
    try:
        if reader.start():
            print("Reading data for 10 seconds...")
            
            for i in range(100):  # Read for 10 seconds at 10Hz
                sample = reader.get_latest_sample()
                if sample:
                    print(f"EEG: {sample['value']:.2f} at {sample['timestamp']:.0f}ms")
                time.sleep(0.1)
        else:
            print("Failed to start Pico reader. Check connection and port.")
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        reader.stop()
