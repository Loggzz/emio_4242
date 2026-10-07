# emio_4242

This repository contains small Python programs for connecting to and testing an
EMIO soft robot. It uses `emioapi` to control the four Dynamixel motors and read
the Intel RealSense depth camera.

## Files

| File | Purpose |
| --- | --- |
| `move.py` | Interactive motor-control experiment. It asks how each leg is mounted, connects to the motors and camera, tracks a green end-effector marker, estimates a local 3-by-4 Jacobian by perturbing each motor by 3 degrees, and then predicts and measures the end-effector position for user-approved angle commands. It restores the reference pose after calibration and disconnects the hardware on exit. |
| `camera_view.py` | Minimal RealSense camera example. It enables marker tracking, point-cloud computation, and preview windows; captures one update; and prints the detected marker positions and point-cloud shape. |
| `test.py` | Basic full-device connection test. It connects to the first available EMIO motor controller and matching camera, prints the status, commands all four motors to 90 degrees, and disconnects. |
| `main.py` | Placeholder application entry point that currently prints `Hello from emio!`; it does not connect to the robot. |
| `pyproject.toml` | Python project metadata. It declares Python 3.12 or newer and the `emioapi` dependency. |
| `uv.lock` | Generated `uv` lockfile containing the exact resolved versions and hashes of `emioapi` and its transitive dependencies. |
| `.python-version` | Selects Python 3.12 for tools that support Python version files. |
| `.gitignore` | Excludes virtual environments, Python caches, build output, test/coverage caches, and editor settings from Git. |
| `.vscode/settings.json` | Local VS Code configuration that adds the ROS 2 Jazzy Python package directory to autocomplete and static analysis. This directory is ignored by Git. |
| `README.md` | This project overview and file reference. |

Generated directories such as `.venv/` and `__pycache__/` contain the local
Python environment and bytecode caches; they are not project source files.

## Safety

Both `move.py` and `test.py` can move the physical robot. `move.py` requires
confirmation before calibration and before each requested command. `test.py`
does not ask for confirmation and immediately requests a 90-degree position on
all four motors after connecting. Keep hands and obstacles clear before running
either script.
