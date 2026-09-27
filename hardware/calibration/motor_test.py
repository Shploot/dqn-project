"""
motor_test.py

Tests both motors on the TB6612FNG driver, wired as:
    AIN1 -> GPIO 5      AIN2 -> GPIO 6      PWMA -> GPIO 12
    BIN1 -> GPIO 20     BIN2 -> GPIO 21     PWMB -> GPIO 13
    STBY -> tied to 3.3V directly (no GPIO needed, always enabled)

Run on the Pi:
    python3 motor_test.py
"""

import time

from gpiozero import Device, Motor
from gpiozero.pins.lgpio import LGPIOFactory

Device.pin_factory = LGPIOFactory()

# gpiozero's Motor class handles a direction pin pair + PWM speed pin
# together, matching exactly how the TB6612FNG expects to be driven.
motor_a = Motor(forward=6, backward=5, pwm=True, enable=12)  # swapped: this motor's wiring was reversed
motor_b = Motor(forward=20, backward=21, pwm=True, enable=13)


def run_test(name, motor):
    print(f"\n--- Testing {name} ---")

    print("  Forward at 50% speed for 1.5s...")
    motor.forward(0.5)
    time.sleep(1.5)
    motor.stop()
    time.sleep(0.5)

    print("  Backward at 50% speed for 1.5s...")
    motor.backward(0.5)
    time.sleep(1.5)
    motor.stop()
    time.sleep(0.5)

    print("  Forward at 100% speed for 1s...")
    motor.forward(1.0)
    time.sleep(1.0)
    motor.stop()

    print(f"  {name} test complete.")


try:
    print("Motor test starting. Make sure the robot has clearance to move!")
    print("Press Ctrl+C at any time to stop immediately.\n")
    time.sleep(9)  # cooldown to grab the robot / clear space if needed

    run_test("Motor A (left)", motor_a)
    time.sleep(1)
    run_test("Motor B (right)", motor_b)

    print("\nSUCCESS: both motors responded. If either didn't move or spun the")
    print("wrong direction, check that motor's wiring polarity (swap the two")
    print("motor output wires on the driver to reverse its direction in hardware,")
    print("or just swap which pin is 'forward' vs 'backward' in this script).")

except KeyboardInterrupt:
    print("\nStopped by user.")
finally:
    motor_a.stop()
    motor_b.stop()
