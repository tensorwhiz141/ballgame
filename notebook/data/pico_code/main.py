import machine
import time
import sys
import select

# EOG sensor connected to ADC pin (GP26)
eog_sensor = machine.ADC(26)
led = machine.Pin("LED", machine.Pin.OUT)

while True:
    # Read analog EOG value (0–65535)
    eog_value = eog_sensor.read_u16()

    # Send value to PC
    print(eog_value)
    
    # Check if PC sent back blink signal
    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
        response = sys.stdin.readline().strip()
        if response == "BLINK":
            led.on()
        elif response == "REST":
            led.off()
    
    time.sleep(0.1)  # Sampling delay
