import math
import time
from dataclasses import dataclass

import numpy as np
from emioapi import EmioAPI


MOTOR_COUNT = 4
MIN_ANGLE = -math.pi
MAX_ANGLE = math.pi
CALIBRATION_STEP = math.radians(3.0)
MOTOR_TIMEOUT = 10.0
TRACKER_TIMEOUT = 5.0
TRACKER_SAMPLES = 15

# Recommended extended configuration in the official EMIO lab.
DEFAULT_COUPLINGS = ("ccw", "cw", "ccw", "cw")
COUPLING_NAMES = {
    "cw": "clockwisedown",
    "ccw": "counterclockwisedown",
}


@dataclass
class LinearizedKinematics:
    """Local soft-robot model: y = y0 + J_SR @ (u - u0)."""

    reference_angles: np.ndarray
    reference_effector: np.ndarray
    jacobian: np.ndarray
    couplings: tuple[str, ...]

    def predict(self, joint_angles) -> np.ndarray:
        angles = np.asarray(joint_angles, dtype=float)
        if angles.shape != (MOTOR_COUNT,):
            raise ValueError("Exactly four joint angles are required.")
        return self.reference_effector + self.jacobian @ (
            angles - self.reference_angles
        )

    def rebase(self, joint_angles, effector_position) -> None:
        """Anchor later local predictions to the latest measured robot pose."""
        self.reference_angles = np.asarray(joint_angles, dtype=float)
        self.reference_effector = np.asarray(effector_position, dtype=float)


def choose_couplings() -> tuple[str, ...]:
    print("\nChoose the physical mounting of each leg.")
    print("  cw   = clockwise down")
    print("  ccw  = counterclockwise down")
    print("Press Enter to accept the shown default.\n")

    couplings = []
    for motor, default in enumerate(DEFAULT_COUPLINGS):
        while True:
            choice = input(f"Motor {motor} coupling [cw/ccw] ({default}): ").strip().lower()
            choice = choice or default
            if choice in COUPLING_NAMES:
                couplings.append(choice)
                break
            print("Please enter 'cw' or 'ccw'.")

    return tuple(couplings)


def validate_angles(angles) -> np.ndarray:
    values = np.asarray(angles, dtype=float)
    if values.shape != (MOTOR_COUNT,):
        raise ValueError("Enter exactly four motor angles.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Motor angles must be finite numbers.")
    if np.any(values < MIN_ANGLE) or np.any(values > MAX_ANGLE):
        raise ValueError("Each motor angle must be between -180 and 180 degrees.")
    return values


def command_angles(motors, angles) -> np.ndarray:
    values = validate_angles(angles)
    motors.angles = values.tolist()
    return values


def wait_for_motors(motors, timeout: float = MOTOR_TIMEOUT) -> None:
    # Give the motion flag time to change after sending a new goal.
    time.sleep(0.15)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not any(bool(value) for value in motors.moving):
            time.sleep(0.25)
            return
        time.sleep(0.05)
    raise TimeoutError("The motors did not settle before the timeout.")


def measure_effector(
    camera,
    expected_position=None,
    samples: int = TRACKER_SAMPLES,
    timeout: float = TRACKER_TIMEOUT,
) -> np.ndarray:
    """Return the median green-marker position in EMIO coordinates (mm)."""
    expected = (
        None
        if expected_position is None
        else np.asarray(expected_position, dtype=float)
    )
    positions = []
    deadline = time.monotonic() + timeout

    while len(positions) < samples and time.monotonic() < deadline:
        camera.update()
        trackers = camera.trackers_pos
        if not trackers:
            continue

        points = np.asarray(trackers, dtype=float)
        if expected is None:
            # The lab setup uses one green marker on the center connector.
            if len(points) != 1:
                continue
            point = points[0]
        else:
            point = points[np.argmin(np.linalg.norm(points - expected, axis=1))]
        positions.append(point)

    if len(positions) < samples:
        raise RuntimeError(
            "Could not track one green end-effector marker. "
            "Check marker visibility and camera HSV calibration."
        )
    return np.median(np.asarray(positions), axis=0)


def calibrate_local_kinematics(emio, couplings) -> LinearizedKinematics:
    """Estimate J_SR with a small finite difference on each physical motor."""
    reference_angles = validate_angles(emio.motors.angles)
    baseline = measure_effector(emio.camera)
    jacobian = np.zeros((3, MOTOR_COUNT), dtype=float)

    print("\nCalibrating the local soft-robot Jacobian...")
    print(f"Reference angles (deg): {np.degrees(reference_angles).round(2).tolist()}")
    print(f"Reference effector (mm): {baseline.round(2).tolist()}")

    try:
        for motor in range(MOTOR_COUNT):
            step = CALIBRATION_STEP
            if reference_angles[motor] + step > MAX_ANGLE:
                step = -CALIBRATION_STEP

            perturbed_angles = reference_angles.copy()
            perturbed_angles[motor] += step

            print(f"  Perturbing motor {motor} by {math.degrees(step):+.1f} deg...")
            command_angles(emio.motors, perturbed_angles)
            wait_for_motors(emio.motors)
            displaced = measure_effector(emio.camera, expected_position=baseline)

            command_angles(emio.motors, reference_angles)
            wait_for_motors(emio.motors)
            returned = measure_effector(emio.camera, expected_position=baseline)

            # Average the before/after reference to reduce slow drift and hysteresis.
            local_baseline = 0.5 * (baseline + returned)
            jacobian[:, motor] = (displaced - local_baseline) / step
            baseline = returned
    finally:
        # Always request the original pose if calibration is interrupted.
        command_angles(emio.motors, reference_angles)
        wait_for_motors(emio.motors)

    final_reference = measure_effector(emio.camera, expected_position=baseline)
    return LinearizedKinematics(
        reference_angles=reference_angles,
        reference_effector=final_reference,
        jacobian=jacobian,
        couplings=tuple(couplings),
    )


def print_model(model: LinearizedKinematics) -> None:
    print("\nPhysical leg configuration:")
    for motor, coupling in enumerate(model.couplings):
        print(f"  Motor {motor}: {COUPLING_NAMES[coupling]}")

    print("\nEstimated J_SR = W_ea W_aa^-1 (mm/rad):")
    print(np.array2string(model.jacobian, precision=3, suppress_small=True))
    print(f"Model origin y0 (mm): {model.reference_effector.round(2).tolist()}")


def prompt_target_angles() -> np.ndarray | None:
    while True:
        value = input(
            "\nEnter four target angles in degrees, or 'q' to quit: "
        ).strip()
        if value.lower() in {"q", "quit", "exit"}:
            return None

        try:
            degrees = [float(item) for item in value.replace(",", " ").split()]
            return validate_angles(np.radians(degrees))
        except ValueError as error:
            print(f"Invalid target: {error}")


def control_loop(emio, model: LinearizedKinematics) -> None:
    while True:
        target_angles = prompt_target_angles()
        if target_angles is None:
            return

        delta_degrees = np.degrees(target_angles - model.reference_angles)
        if np.max(np.abs(delta_degrees)) > 15.0:
            print(
                "Warning: this is a local linear model; a change above 15 degrees "
                "may have a large prediction error."
            )

        predicted = model.predict(target_angles)
        print(f"Predicted end effector [x, y, z] mm: {predicted.round(2).tolist()}")

        if input("Send this command to the robot? [y/N]: ").strip().lower() != "y":
            print("Command skipped.")
            continue

        command_angles(emio.motors, target_angles)
        wait_for_motors(emio.motors)
        measured = measure_effector(emio.camera, expected_position=predicted)
        actual_angles = validate_angles(emio.motors.angles)

        print(f"Measured end effector [x, y, z] mm:  {measured.round(2).tolist()}")
        print(f"Prediction error [x, y, z] mm:     {(measured - predicted).round(2).tolist()}")

        # Keep subsequent commands local to the latest measured pose.
        model.rebase(actual_angles, measured)


def main() -> None:
    couplings = choose_couplings()
    print("\nMount the four legs exactly as selected and attach one green marker")
    print("to the center of the connector. Calibration moves one motor at a time")
    print("by 3 degrees. Keep hands and obstacles clear of the robot.")
    if input("Type MOVE to connect and begin calibration: ").strip() != "MOVE":
        print("Cancelled without moving the robot.")
        return

    emio = EmioAPI(multiprocess_camera=False)
    emio.camera.track_markers = True

    try:
        if not emio.connectToEmioDevice():
            print("Failed to connect to the EMIO motors and camera.")
            return

        print("Connected to EMIO.")
        model = calibrate_local_kinematics(emio, couplings)
        print_model(model)
        control_loop(emio, model)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except (RuntimeError, TimeoutError, ValueError) as error:
        print(f"Stopped: {error}")
    finally:
        if emio.motors.is_connected or emio.camera.is_running:
            emio.disconnect()


if __name__ == "__main__":
    main()
