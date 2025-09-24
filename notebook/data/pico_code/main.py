# This code runs on the Raspberry Pi Pico
import machine
import time

# Set up the Analog-to-Digital Converter on GPIO Pin 26 (which is ADC0)
# This is where your EMG sensor's signal pin should be connected.
adc = machine.ADC(26)

# Main loop to continuously read and send data
while True:
    # Read the 16-bit analog value (an integer between 0 and 65535)
    sensor_value = adc.read_u16()

    # Print the value to the USB serial port so the PC can read it
    print(sensor_value)

    # Wait for 10 milliseconds. This creates a 100 Hz sampling rate.
    # You can adjust this delay if needed.
    time.sleep(0.01)