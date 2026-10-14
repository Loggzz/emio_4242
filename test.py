import math
from emioapi import EmioAPI


# Create an EmioAPI instance
emio = EmioAPI(multiprocess_camera=False)

# Connect to the first available Emio device
if emio.connectToEmioDevice():
    print("Connected to Emio device.")

    # Print device status
    emio.printStatus()

    # Example: Move all motors to 90 degrees (PI/2 radians)
    target_angles = [math.pi/2] * 4
    emio.motors.angles = target_angles

    # Disconnect when done
    emio.disconnect()
else:
    print("Failed to connect to Emio device.")