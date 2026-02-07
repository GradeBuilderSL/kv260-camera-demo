# Kria Camera Demo

A real-time camera demonstration for AMD Xilinx Kria KV260 Vision AI Starter Kit with Intel RealSense depth camera integration. This demo captures and displays both color and depth streams side-by-side, leveraging the Kria's DPU (Deep Learning Processing Unit) capabilities.

## Features

- Real-time Intel RealSense camera stream capture (640x480 @ 30fps)
- Simultaneous color and depth image visualization
- DPU overlay integration for AI acceleration
- Color-mapped depth visualization using OpenCV

## Hardware Requirements

- **AMD Xilinx Kria KV260 Vision AI Starter Kit**
- **Intel RealSense Depth Camera** (D400 series recommended)
- USB 3.0 connection for camera
- Network connection for remote deployment

## Software Requirements

- PYNQ framework with DPU support
- Python 3.x
- DPU bitstream (`dpu.bit`)

## Setup

### On Development Machine

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd kria-camera-demo
   ```

### On Kria Board

1. Ensure PYNQ environment is installed and activated:
   ```bash
   source /usr/local/share/pynq-venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip3 install -r requirements.txt
   ```

3. Ensure the DPU bitstream (`dpu.bit`) is available in the project directory.

## Deployment

### Deploy to Kria (from development machine)

Use the VS Code task or run manually:

```bash
rsync -avz --exclude=venv --exclude=.git --exclude=__pycache__ \
  ./ ubuntu@192.168.100.8:~/kria-camera-demo/
```

## Usage

### Run Locally on Kria

```bash
ssh ubuntu@192.168.100.8
cd ~/kria-camera-demo
sudo /usr/local/share/pynq-venv/bin/python3 kria-camera-demo.py
```

### Run via VS Code Task

Use the "Run on Kria" task which automatically deploys and executes the demo.

### Controls

- **q**: Quit the application
- The display shows color image on the left and color-mapped depth on the right

## Project Structure

```
kria-camera-demo/
├── kria-camera-demo.py    # Main application
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── dpu.bit                # DPU overlay bitstream (required)
```

## Dependencies

- **pyrealsense2**: Intel RealSense SDK for camera control
- **opencv-python**: Image processing and visualization
- **numpy**: Numerical operations
- **pynq_dpu**: PYNQ DPU overlay support

## Troubleshooting

- **Camera not detected**: Ensure RealSense camera is connected via USB 3.0
- **Permission denied**: Run with `sudo` as camera access requires elevated privileges
- **DPU overlay error**: Verify `dpu.bit` file exists in the project directory
- **Module not found**: Activate PYNQ virtual environment before running

## Network Configuration

Default Kria board IP address: `192.168.100.8`

Update the IP address in `.vscode/tasks.json` if your board uses a different address.

## License

[Add your license here]
