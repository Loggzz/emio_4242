"""Print EMIO's current joint angles once, without changing motor settings."""

import math

from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler
from serial.tools import list_ports


def main() -> int:
    ports = [
        port.device
        for port in list_ports.comports()
        if "FTDI" in " ".join(
            (port.manufacturer or "", port.description or "", port.serial_number or "")
        )
    ]
    if not ports:
        print("No EMIO motor controller found. Check the USB connection.")
        return 1

    port = PortHandler(ports[0])
    packet = PacketHandler(2.0)
    try:
        if not port.openPort() or not port.setBaudRate(1_000_000):
            raise RuntimeError(f"Could not open {ports[0]} at 1,000,000 baud.")

        angles = []
        for motor_id in range(4):
            # Read Present Position only; do not enable torque or set goals.
            pulses, result, error = packet.read4ByteTxRx(port, motor_id, 132)
            if result != COMM_SUCCESS:
                raise RuntimeError(f"Motor {motor_id}: {packet.getTxRxResult(result)}")
            if error:
                raise RuntimeError(f"Motor {motor_id}: {packet.getRxPacketError(error)}")
            if pulses >= 2**31:
                pulses -= 2**32
            # EMIO uses 2048 pulses as zero, with the reversed angle convention.
            angles.append((2048 - pulses) * math.tau / 4096)

        print("Motor IDs:   [0, 1, 2, 3]")
        print("Angles (rad):", [round(angle, 4) for angle in angles])
        print("Angles (deg):", [round(math.degrees(angle), 2) for angle in angles])
        return 0
    except (RuntimeError, OSError) as error:
        print(f"Could not read joint poses: {error}")
        return 1
    finally:
        if port.is_open:
            port.closePort()


if __name__ == "__main__":
    raise SystemExit(main())
