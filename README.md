# Kria Camera Demo

A real-time camera demonstration for AMD Xilinx Kria KV260 Vision AI Starter Kit with Intel RealSense depth camera integration. This demo captures and displays both color and depth streams side-by-side, leveraging the Kria's DPU (Deep Learning Processing Unit) capabilities.

## Features

- **Real-time Object Classification**: MobileNet V2 inference on live RealSense camera feed
- **Dual FPS Monitoring**: Displays both overall frame rate and DPU inference speed
- **System Monitoring**: Live display of power consumption, temperatures, CPU utilization, and memory usage
- **Top-5 Predictions**: Shows top 5 classification results with confidence scores
- **Smart Text Rendering**: Auto-contrast text overlays with semi-transparent backgrounds for maximum readability
- **Organized Display**: Sectioned interface with "Classification" and "System Info" headers
- **Depth Visualization**: Color-mapped depth stream using OpenCV (640x480 @ 30fps)
- **Modular Architecture**: Clean separation of concerns across multiple modules

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

### Command-Line Options

```bash
python3 kria-camera-demo.py [OPTIONS]

Options:
  -m, --model-dir PATH   Path to model directory containing meta.json
                        (default: models/mobilenet_v2)
  -b, --dpu-bit PATH    Path to DPU bitstream file (default: dpu.bit)
  -l, --labels PATH     Path to class labels file (default: words.txt)
  -h, --help           Show help message
```

### Model Directory Structure

The model directory must contain:
- `meta.json` - Model metadata with the following structure:
  ```json
  {
      "lib": "libvart-dpu-runner.so",
      "filename": "mobilenet_v2.xmodel",
      "kernel": ["subgraph_263"],
      "target": "DPUCZDX8G_ISA1_B4096"
  }
  ```
- `<model_name>.xmodel` - The compiled DPU model file (specified in meta.json filename field)
- `<model_name>.prototxt` - Preprocessing metadata with normalization parameters

### Using a Different Model

```bash
# Use a custom model directory
sudo /usr/local/share/pynq-venv/bin/python3 kria-camera-demo.py -m models/resnet50

# Specify all paths
sudo /usr/local/share/pynq-venv/bin/python3 kria-camera-demo.py \
  -m models/custom_model \
  -b custom_dpu.bit \
  -l custom_labels.txt
```

### Run via VS Code Task

Use the "Run on Kria" task which automatically deploys and executes the demo.

### Controls

- **q**: Quit the application
- The display shows color image on the left and color-mapped depth on the right

## Project Structure

The codebase has been refactored into modular components for better maintainability:

```
kria-camera-demo/
├── kria-camera-demo.py      # Main entry point with visualization loop
├── camera_demo/             # Core package modules
│   ├── __init__.py          # Package initialization and exports
│   ├── kria_camera_demo.py  # Core KriaCameraDemo class
│   ├── utils.py             # General utilities (privileges, config parsing, labels)
│   ├── preprocessing.py     # Image preprocessing pipeline for DPU
│   ├── visualization.py     # Display and text rendering helpers
│   └── platform_monitor.py  # System statistics monitoring (PlatformMonitor class)
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── models/                 # Model files (MobileNet V2)
│   └── mobilenet_v2/
│       ├── mobilenet_v2.xmodel
│       └── mobilenet_v2.prototxt
├── words.txt               # ImageNet class labels
└── dpu.bit                 # DPU overlay bitstream (required)
```

### Module Descriptions

- **`kria-camera-demo.py`**: Main entry point that initializes the application and runs the display loop
- **`camera_demo/`**: Python package containing all core modules
  - **`kria_camera_demo.py`**: `KriaCameraDemo` class orchestrating camera, DPU inference, and monitoring
  - **`utils.py`**: Privilege checking, prototxt parsing, and label loading utilities
  - **`preprocessing.py`**: Image preprocessing (resize, crop, normalize) for MobileNet V2
  - **`visualization.py`**: Smart text rendering with auto-contrast and semi-transparent backgrounds
  - **`platform_monitor.py`**: Background thread monitoring power, temperature, CPU, and memory stats

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
