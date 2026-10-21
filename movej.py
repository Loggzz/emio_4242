"""Move the physical EMIO robot through a predefined picking sequence."""

import time

from emioapi import EmioAPI


MOVE_TIMEOUT_SECONDS = 10.0

POSE_SEQUENCE = (
    ("home", (1.96, -1.16, 1.96, -1.16)),
    ("hover", (2.2841, -0.8774, 1.8101, -0.7885)),
    ("approach", (0.4571, -0.5016, 0.1979, -0.4203)),
    ("pick", (0.069, -0.7271, -0.1626, -0.6412)),
    ("close_hover", (1.0738, -1.7196, 0.7747, -1.5907)),
    ("pose1_hover", (0.8943, -2.1721, -0.5538, -0.3359)),
    ("pose1_approach", (0.6765, -2.0233, -0.7578, -0.2301)),
    ("pose1_release", (1.2916, -1.4772, -0.4034, 0.0399)),
    ("hover", (2.2841, -0.8774, 1.8101, -0.7885)),
    ("approach", (0.4571, -0.5016, 0.1979, -0.4203)),
    ("pick", (0.069, -0.7271, -0.1626, -0.6412)),
    ("close_hover", (1.0738, -1.7196, 0.7747, -1.5907)),
    ("pose2_hover", (0.2255, -2.3823, 0.566, -0.3605)),
    ("pose2_approach", (-0.4679, -1.669, -0.1795, 0.1595)),
    ("pose2_release", (-0.0966, -1.3192, 0.2408, 0.3927)),
    ("pose2_open_hover", (0.6872, -1.7503, 1.0999, 0.0982)),
    ("hover", (2.2841, -0.8774, 1.8101, -0.7885)),
    ("approach", (0.4571, -0.5016, 0.1979, -0.4203)),
    ("pick", (0.069, -0.7271, -0.1626, -0.6412)),
    ("close_hover", (1.0738, -1.7196, 0.7747, -1.5907)),
    ("pose3_hover", (-0.3206, -0.4786, 0.7655, -2.2841)),
    ("pose3_approach", (-0.6765, -0.2194, 0.3206, -1.8454)),
    ("pose3_release", (-0.3758, -0.0353, 0.767, -1.4818)),
    ("pose3_open_hover", (0.2224, -0.3114, 1.6076, -1.939)),
    ("hover", (2.2841, -0.8774, 1.8101, -0.7885)),
    ("approach", (0.4571, -0.5016, 0.1979, -0.4203)),
    ("pick", (0.069, -0.7271, -0.1626, -0.6412)),
    ("close_hover", (1.0738, -1.7196, 0.7747, -1.5907)),
    ("pose4_hover", (0.3022, -0.6427, -0.9603, -2.2028)),
    ("pose4_approach", (0.3022, -0.6427, -0.9603, -2.2028)),
    ("pose4_release", (1.0937, -0.2163, -0.4295, -1.4251)),
    ("pose4_open_hover", (1.9727, -0.3083, 0.3344, -2.1491)),
    ("hover", (2.2841, -0.8774, 1.8101, -0.7885)),
    ("approach", (0.4571, -0.5016, 0.1979, -0.4203)),
    ("pick", (0.069, -0.7271, -0.1626, -0.6412)),
    ("close_hover", (1.0738, -1.7196, 0.7747, -1.5907)),
    ("pose5_hover", (-0.112, -1.6245, 1.1781, -1.4634)),
    ("pose5_approach", (-0.6949, -1.0216, 0.4019, -0.8912)),
    ("pose5_release", (-0.2454, -0.6044, 1.0646, -0.4939)),
    ("pose5_open_hover", (0.4019, -0.8529, 2.1138, -0.7348)),
    ("hover", (2.2841, -0.8774, 1.8101, -0.7885)),
    ("approach", (0.4571, -0.5016, 0.1979, -0.4203)),
    ("pick", (0.069, -0.7271, -0.1626, -0.6412)),
    ("close_hover", (1.0738, -1.7196, 0.7747, -1.5907)),
    ("pose6_hover", (0.856, -0.5323, 0.5829, -2.8072)),
    ("pose6_approach", (0.1365, -0.1197, -0.092, -1.9543)),
    ("pose6_release", (0.5614, 0.1396, 0.2991, -1.488)),
    ("pose6_open_hover", (1.2118, -0.0568, 0.8897, -1.7963)),
)

def wait_for_motors(motors, timeout: float = MOVE_TIMEOUT_SECONDS) -> None:
    time.sleep(5)

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
        emio.motors.max_velocity = [25, 25, 25, 25]

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
