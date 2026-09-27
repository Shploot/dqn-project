"""
gpio_check.py

Run this FIRST, before testing motors or sensors. It confirms:
  1. The gpiozero library + Pi 5 GPIO backend (lgpio) are installed
     and working at all.
  2. The Pi can toggle a GPIO pin without errors.

If this fails, nothing else (motors, sensors) will work either --
fix this first before moving on.

Run on the Pi (via your SSH/VS Code terminal), not on your PC:
    python3 gpio_check.py
"""

import time

try:
    from gpiozero import LED
    from gpiozero.pins.lgpio import LGPIOFactory
    from gpiozero import Device

    Device.pin_factory = LGPIOFactory()
    print("gpiozero + lgpio backend loaded successfully.")
except ImportError as e:
    print("ERROR: gpiozero (or its lgpio backend) is not installed.")
    print("Fix with: pip3 install gpiozero lgpio --break-system-packages")
    print(f"Details: {e}")
    raise SystemExit(1)

# GPIO 17 is a common safe "scratch" pin for testing -- change if you
# have something else already wired there.
TEST_PIN = 17

print(f"\nToggling GPIO {TEST_PIN} on/off 5 times...")
print("(If you have an LED + resistor on this pin, watch it blink.")
print(" If not, this just confirms no errors occur -- that's enough.)\n")

try:
    pin = LED(TEST_PIN)
    for i in range(5):
        pin.on()
        print(f"  [{i+1}/5] Pin {TEST_PIN} -> HIGH")
        time.sleep(0.3)
        pin.off()
        print(f"  [{i+1}/5] Pin {TEST_PIN} -> LOW")
        time.sleep(0.3)
    pin.close()
    print("\nSUCCESS: GPIO is working correctly. Safe to move on to motor/sensor tests.")
except Exception as e:
    print(f"\nERROR: GPIO toggle failed: {e}")
    print("Common causes: wrong pin number, permissions issue, or pin already in use.")
    raise SystemExit(1)