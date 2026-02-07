# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python application for demonstrating Intel RealSense camera capabilities on AMD Xilinx Kria hardware. The project uses a remote deployment workflow where development happens locally but execution occurs on the Kria board at `192.168.100.8`.

## Development Workflow

### Local Setup
```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Remote Deployment to Kria Board

The project is configured for remote deployment to a Kria board at `192.168.100.8`. Use VSCode tasks or manual commands:

**Deploy to Kria:**
```bash
rsync -avz --exclude=venv --exclude=.git --exclude=__pycache__ ./ ubuntu@192.168.100.8:~/kria-camera-demo/
```

**Run on Kria:**
```bash
ssh ubuntu@192.168.100.8 "cd ~/kria-camera-demo && sudo /usr/local/share/pynq-venv/bin/python3 kria-camera-demo.py"
```

**Note:** Root permissions are required for PYNQ DPU overlay to access hardware MMIO registers.

**Install dependencies on Kria:**
```bash
ssh ubuntu@192.168.100.8 "cd ~/kria-camera-demo && pip3 install -r requirements.txt"
```

**Deploy and run (combined):**
Use the VSCode task "Run on Kria (192.168.100.8)" which automatically deploys first, then executes.

## Application Architecture

### Main Application: kria-camera-demo.py

The primary application captures and visualizes RealSense camera streams:

1. **Pipeline Setup**: Configures Intel RealSense pipeline with depth (640x480, z16, 30fps) and color (640x480, BGR8, 30fps) streams
2. **Frame Capture**: Continuously reads coherent depth and color frame pairs
3. **Visualization**: Converts depth to colormap (using WINTER colormap) and displays side-by-side with color feed via OpenCV
4. **Exit**: Press 'q' to quit the application

### Dependencies

- `pyrealsense2`: Intel RealSense SDK for camera interface
- `opencv-python`: Image processing and visualization
- `numpy`: Array operations for frame data

### Remote Execution Context

The application requires:
- Physical RealSense camera connected to the Kria board
- X11 display forwarding configured (DISPLAY=192.168.100.8:0)
- SSH access to ubuntu@192.168.100.8
- Root permissions (sudo) for PYNQ hardware access

## File Structure

- `kria-camera-demo.py`: Main RealSense camera application (actively used)
- `main.py`: Placeholder/stub file (minimal implementation)
- `requirements.txt`: Python dependencies
- `.vscode/tasks.json`: Deployment and remote execution tasks
- `.vscode/launch.json`: Debug configurations with remote display settings
