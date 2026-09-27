"""
sensor_test.py

Tests all 3 HC-SR04 ultrasonic sensors, wired as:
    Left sensor:   Echo -> GPIO 4    Trig -> GPIO 17
    Middle sensor: Echo -> GPIO 18   Trig -> GPIO 27
    Right sensor:  Echo -> GPIO 22   Trig -> GPIO 23

Prints continuous distance readings (in cm) from all 3 sensors so you
can wave a hand/object in front of each one and confirm the readings
change sensibly.

Run on the Pi:
    python3 sensor_test.py
Press Ctrl+C to stop.
"""

import time

from gpiozero import Device, DistanceSensor
from gpiozero.pins.lgpio import LGPIOFactory

Device.pin_factory = LGPIOFactory()

# max_distance in meters -- HC-SR04's practical reliable range is
# about 4m, but readings get noisier near the edge, so we cap it a
# bit under datasheet spec for more reliable readings.
MAX_DISTANCE_M = 3.0

left_sensor = DistanceSensor(echo=4, trigger=17, max_distance=MAX_DISTANCE_M)
middle_sensor = DistanceSensor(echo=18, trigger=27, max_distance=MAX_DISTANCE_M)
right_sensor = DistanceSensor(echo=22, trigger=23, max_distance=MAX_DISTANCE_M)


def read_cm(sensor):
    # gpiozero's DistanceSensor.distance returns meters (0.0 to 1.0
    # scale relative to max_distance); convert to cm for readability.
    return sensor.distance * 100


try:
    print("Reading all 3 sensors continuously. Each reading prints on its own")
    print("line, so you can walk away, wave your hands around the robot, then")
    print("come back and scroll up to see what changed over that time.")
    print("Press Ctrl+C to stop.\n")

    start_time = time.time()

    while True:
        left_cm = read_cm(left_sensor)
        middle_cm = read_cm(middle_sensor)
        right_cm = read_cm(right_sensor)
        elapsed = time.time() - start_time

        print(
            f"[{elapsed:6.1f}s] Left: {left_cm:6.1f} cm   |   "
            f"Middle: {middle_cm:6.1f} cm   |   "
            f"Right: {right_cm:6.1f} cm"
        )
        time.sleep(0.5)

except KeyboardInterrupt:
    print("\n\nStopped by user.")
    print("If all 3 readings changed sensibly as you moved objects in front")
    print("of them, sensors are working correctly and matched to the right")
    print("physical position (left/middle/right).")