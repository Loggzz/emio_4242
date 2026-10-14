"""Move the physical EMIO robot through a predefined picking sequence."""

import time

from emioapi import EmioAPI


MOVE_TIMEOUT_SECONDS = 10.0

POSE_SEQUENCE = (
    ("home", (1.53, -1.59, 1.79, -1.33)),
    ("hover", (1.14, -1.06, 1.32, -1.13)),
    ("approach", (0.11, -0.58, 0.24, -0.63)),
    ("pick", (-0.36, -0.86, -0.25, -0.93)),
)

def wait_for_motors(motors, timeout: float = MOVE_TIMEOUT_SECONDS) -> None:
    """Wait until all motors stop, or raise if the move takes too long."""
    time.sleep(0.15)  # Allow the moving flags to update after a new command.
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        if not any(bool(value) for value in motors.moving):
            return
        time.sleep(0.05)

    raise TimeoutError("The motors did not reach the target pose in time.")


def move_j(joint_angles, emio: EmioAPI) -> None:
    """Move all four motors to the requested joint angles in radians."""
    emio.motors.angles = list(joint_angles)
    wait_for_motors(emio.motors)


def main() -> None:
    emio = EmioAPI(multiprocess_camera=False)

    if not emio.connectToEmioDevice():
        print("Failed to connect to Emio device.")
        return

    print("Connected to Emio device.")

    try:
        emio.motors.max_velocity = [8,8,8,8]

        emio.printStatus()

        for name, joint_angles in POSE_SEQUENCE:
            print(f"Moving to {name} pose: {list(joint_angles)}")
            move_j(joint_angles, emio)

        print("Picking sequence complete.")
    except (TimeoutError, ValueError) as error:
        print(f"Motion stopped: {error}")
    except KeyboardInterrupt:
        print("\nMotion stopped by user.")
    finally:
        emio.disconnect()
        print("Disconnected from Emio device.")


if __name__ == "__main__":
    main()
