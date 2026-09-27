"""
odometry.py

Real encoder-based position tracking, replacing the open-loop dead-
reckoning guesswork in deploy.py's REAL_MOVE_SPEED_M / REAL_TURN_SPEED_RAD
placeholders. Uses quadrature encoders on each TT motor (DFRobot
FIT0450: 960 pulses per output-shaft revolution after the 120:1
gearbox) plus differential-drive kinematics to compute the robot's
actual (x, y, heading) from real wheel rotation -- no calibration,
no drift from guessed speeds.

Wiring (confirmed):
    Left motor:  Phase A -> GPIO 24   Phase B -> GPIO 25
    Right motor: Phase A -> GPIO 19   Phase B -> GPIO 26

Run standalone to test:
    python3 odometry.py
    (spin the wheels by hand or run motor_test.py in another terminal
    and watch the printed pose update)
"""

import math
import time

from gpiozero import Device
from gpiozero.pins.lgpio import LGPIOFactory
import lgpio

Device.pin_factory = LGPIOFactory()

# ------------------------------------------------------------------ #
# Robot geometry -- measured values
# ------------------------------------------------------------------ #
WHEEL_DIAMETER_M = 0.065
WHEEL_CIRCUMFERENCE_M = math.pi * WHEEL_DIAMETER_M
# DFRobot FIT0450 spec: 8 pulses/rev x 120:1 gearbox = 960, for the ORIGINAL
# single-channel (2x) decode. Doubled here to 1920 to match Encoder's new
# full 4x decoding (both A and B edges counted, not just A) -- otherwise
# every distance/heading value computed from pulse counts would read
# exactly 2x too large now that twice as many pulses are counted for the
# same real rotation.
PULSES_PER_REV = 1920

# Distance between the two wheel centers, used for heading math.
# CAD motor-mount spacing is 184mm, but reverted to the original 0.15
# placeholder: the power-scale calibrations (TURN_LEFT_SCALE,
# FORWARD_LEFT_SCALE, etc.) were all measured via turn_drift_test.py/
# forward_drift_test.py while THIS was the active value, and all three
# clean successful deploy.py runs tonight happened with 0.15 in place.
# Changing to the "more correct" 184mm broke that matched calibration
# set and made real performance worse (more turns, more steps, no
# clean success). 0.15 is the proven, empirically-working value.
WHEEL_TRACK_M = 0.15


# Minimum physically-plausible time between encoder pulses, in
# microseconds. Based on the FIT0450's spec (max ~2560 pulses/sec at
# full 6V no-load speed), any two edges closer together than this are
# essentially impossible from real rotation -- almost certainly
# electrical noise from motor PWM switching corrupting the signal
# (the same class of interference that required median-filtering the
# ultrasonic sensors). Rejecting these glitch edges instead of
# counting them prevents the wildly inconsistent, non-monotonic pulse
# counts that noise otherwise causes.
#
# Halved from the original 150us: that value assumed only A-channel
# edges (2 transitions per quadrature cycle). Now that both A and B
# edges are decoded (4 transitions per cycle, see Encoder below), genuine
# edges arrive roughly twice as often at the same real rotation speed, so
# the same physical speed limit now corresponds to half the old spacing.
MIN_PULSE_INTERVAL_US = 75


class Encoder:
    """Reads a single quadrature encoder (2 pins: A, B) and tracks signed
    pulse count (direction-aware) using full 4x decoding: an interrupt on
    EITHER pin looks at both pins' current levels to determine the new
    2-bit (A,B) state, and a transition table maps (old_state, new_state)
    to a direction.

    This replaced an earlier version that only interrupted on pin A and
    read pin B's level inline inside that callback to infer direction.
    That extra gpio_read() has real latency, and at high rotation speed B
    can flip before it's sampled, misreading the transition -- confirmed
    on real hardware via single_turn_test.py: a turn that visibly rotated
    the robot ~90 degrees was reported as ~0 degrees by odometry, with
    near-zero raw pulse counts on both encoders and 0 rejected pulses (so
    it wasn't the debounce filter either) -- consistent with edges being
    miscounted/cancelling out rather than genuinely absent. Decoding both
    channels directly removes the inline read and its race window, and
    doubles resolution as a side effect (see PULSES_PER_REV).
    """

    # (old_state, new_state) -> direction, where state = (a_level << 1) |
    # b_level. Chosen to exactly reproduce the prior single-channel
    # decode's already-validated sign convention (see the SIGN CONVENTION
    # FIX note below) for every transition it was able to see -- this
    # table just also covers the B-edge transitions that version couldn't
    # see at all, it doesn't change what "positive" means.
    _TRANSITIONS = {
        (0, 1): 1, (1, 3): 1, (3, 2): 1, (2, 0): 1,
        (0, 2): -1, (2, 3): -1, (3, 1): -1, (1, 0): -1,
    }

    def __init__(self, chip_handle, pin_a, pin_b):
        self.chip = chip_handle
        self.pin_a = pin_a
        self.pin_b = pin_b
        self.count = 0
        self.rejected_count = 0  # how many implausible edges got filtered out -- useful for diagnosing noise severity
        self._last_tick = None

        lgpio.gpio_claim_alert(self.chip, pin_a, lgpio.BOTH_EDGES)
        lgpio.gpio_claim_alert(self.chip, pin_b, lgpio.BOTH_EDGES)

        a_level = lgpio.gpio_read(self.chip, pin_a)
        b_level = lgpio.gpio_read(self.chip, pin_b)
        self._state = (a_level << 1) | b_level

        # SIGN CONVENTION FIX (carried over from the single-channel
        # version): confirmed via a real deployment run that commanded
        # "forward" (physically verified moving forward) was producing
        # NEGATIVE encoder counts under the original state table -- the
        # opposite of what the position math needs (forward should be
        # positive x, matching where goals are placed). This also
        # silently inverted HEADING tracking the same way, since heading
        # is computed from the left/right difference. _TRANSITIONS above
        # already reflects the corrected sign.
        self._callback_a = lgpio.callback(self.chip, pin_a, lgpio.BOTH_EDGES, self._on_edge)
        self._callback_b = lgpio.callback(self.chip, pin_b, lgpio.BOTH_EDGES, self._on_edge)

    def _on_edge(self, chip, gpio, level, tick):
        # debounce: reject edges that arrive faster than physically
        # possible -- almost certainly electrical noise, not real
        # rotation. lgpio's tick is a wrapping microsecond counter;
        # this simple check is safe for runs well under ~35 minutes
        # (half the 32-bit wraparound period).
        if self._last_tick is not None:
            elapsed = tick - self._last_tick
            if 0 <= elapsed < MIN_PULSE_INTERVAL_US:
                self.rejected_count += 1
                return
        self._last_tick = tick

        if gpio == self.pin_a:
            a_bit, b_bit = level, self._state & 0b01
        else:
            a_bit, b_bit = (self._state >> 1) & 0b01, level
        new_state = (a_bit << 1) | b_bit

        # unrecognized transitions (same state re-firing, or a skipped
        # state from a genuinely missed edge) can't be signed confidently
        # -- .get(..., 0) leaves the count unchanged for those rather than
        # guessing a direction.
        self.count += self._TRANSITIONS.get((self._state, new_state), 0)
        self._state = new_state

    def close(self):
        self._callback_a.cancel()
        self._callback_b.cancel()


class Odometry:
    """Tracks the robot's (x, y, heading) using both wheel encoders
    and standard differential-drive kinematics."""

    def __init__(self):
        self.chip = lgpio.gpiochip_open(0)
        self.left_encoder = Encoder(self.chip, 24, 25)
        self.right_encoder = Encoder(self.chip, 19, 26)

        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0  # radians

        self._last_left_count = 0
        self._last_right_count = 0

    def _pulses_to_meters(self, pulses):
        revolutions = pulses / PULSES_PER_REV
        return revolutions * WHEEL_CIRCUMFERENCE_M

    def update(self):
        """Call this regularly (e.g. once per control loop step) to
        update the pose estimate based on encoder counts since the
        last call."""
        left_count = self.left_encoder.count
        right_count = self.right_encoder.count

        delta_left_pulses = left_count - self._last_left_count
        delta_right_pulses = right_count - self._last_right_count
        self._last_left_count = left_count
        self._last_right_count = right_count

        delta_left_m = self._pulses_to_meters(delta_left_pulses)
        delta_right_m = self._pulses_to_meters(delta_right_pulses)

        # standard differential-drive odometry update
        #
        # HEADING SIGN -- history worth keeping: single_turn_test.py's real
        # measurements (using the OLD single-channel-A-only Encoder) showed
        # turning LEFT producing NEGATIVE heading and turning RIGHT
        # producing POSITIVE -- backwards from nav_env.py's convention
        # (action 1 "turn left" = heading += turn_speed). That reading led
        # to swapping this formula to (delta_left - delta_right).
        #
        # After Encoder was rewritten to properly decode both A and B edges
        # (see Encoder's docstring -- the old version's inline gpio_read()
        # of B had a race that could misread transitions at real turn
        # speed), re-measuring showed the OPPOSITE: turning LEFT now
        # produces NEGATIVE delta_left_m / POSITIVE delta_right_m, meaning
        # the ORIGINAL (delta_right - delta_left) formula is what actually
        # matches nav_env.py's convention. The earlier swap was compensating
        # for the old encoder's race-condition bug, not a real bug in this
        # formula -- reverted now that the encoder itself reads correctly.
        #
        # The x/y position math (delta_center_m, mid_heading) is unaffected
        # by this either way, since it only depends on the heading's sign
        # being self-consistent, not on which term comes first here.
        delta_center_m = (delta_left_m + delta_right_m) / 2.0
        delta_heading = (delta_right_m - delta_left_m) / (2.0 * WHEEL_TRACK_M)

        # update pose using the midpoint heading for better accuracy
        # over a discrete step (avoids systematic bias vs. using only
        # the starting or ending heading)
        mid_heading = self.heading + delta_heading / 2.0
        self.x += delta_center_m * math.cos(mid_heading)
        self.y += delta_center_m * math.sin(mid_heading)
        self.heading = self._wrap_angle(self.heading + delta_heading)

        return self.x, self.y, self.heading

    @staticmethod
    def _wrap_angle(angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def reset(self):
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0
        self._last_left_count = self.left_encoder.count
        self._last_right_count = self.right_encoder.count

    def close(self):
        self.left_encoder.close()
        self.right_encoder.close()
        lgpio.gpiochip_close(self.chip)


if __name__ == "__main__":
    print("Odometry test. Spin the wheels by hand, or run motor_test.py")
    print("in another terminal, and watch the pose update below.")
    print("Press Ctrl+C to stop.\n")

    odom = Odometry()
    try:
        while True:
            x, y, heading = odom.update()
            heading_deg = math.degrees(heading)
            print(
                f"x: {x:6.3f} m | y: {y:6.3f} m | heading: {heading_deg:6.1f} deg | "
                f"L pulses: {odom.left_encoder.count:6d} | R pulses: {odom.right_encoder.count:6d}",
                end="\r",
            )
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    finally:
        odom.close()