from emioapi import EmioCamera

# Create an instance of EmioCamera
camera = EmioCamera(show=True, track_markers=True, compute_point_cloud=True, configuration="extended")

# Open the camera
if camera.open():
    while camera.is_running:
        # Update camera frames and tracking
        camera.update()

        # Access tracker positions
        positions = camera.trackers_pos
        print("Tracker positions:", positions)

        # Access point cloud data
        pc = camera.point_cloud
        print("Point cloud shape:", pc.shape)

        # Access HSV and mask frames
        hsv = camera.hsv_frame
        mask = camera.mask_frame

        # ... (process frames as needed)

        # For demonstration, break after one iteration