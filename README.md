# Kria Camera Demo & DPU Benchmark

Real-time camera demonstration and DPU performance benchmarking suite for AMD Xilinx Kria KV260 Vision AI Starter Kit.

## Features

### Camera Demo (kria-camera-demo.py)
- **Real-time Object Classification**: MobileNet V2 inference on live RealSense camera feed
- **Dual FPS Monitoring**: Overall frame rate and DPU inference speed
- **System Monitoring**: Power, temperature, CPU utilization, and memory usage
- **Top-5 Predictions**: Classification results with confidence scores
- **Smart Text Rendering**: Auto-contrast overlays with semi-transparent backgrounds
- **Depth Visualization**: Color-mapped depth stream (640x480 @ 30fps)

### DPU Benchmark (utils/benchmark_dpu.py)
- **Synthetic Testing**: Uses random images (no camera required)
- **Vaitrace Instrumentation**: Layer-by-layer DPU profiling
- **Comprehensive Statistics**: Mean, std, min, max, P50, P95, P99
- **Reproducible Results**: Fixed random seed support
- **Warmup Phase**: Accurate performance measurements

### Trace Analyzer (utils/analyze_trace.py)
- **Automatic Analysis**: Runs automatically after `run_vaitrace.sh` completes
- **Per-Layer Stats**: Mean latency, std dev, CoV(%), min/max, efficiency (GOP/s)
- **Timeline Order**: Layers sorted by execution order in the model
- **CSV Export**: Save results with `--csv-out`

## Hardware Requirements

- **AMD Xilinx Kria KV260 Vision AI Starter Kit**
- **Intel RealSense Depth Camera** (D400 series) - for camera demo only
- USB 3.0 connection (for camera)
- Network connection for remote deployment

## Software Requirements

- PYNQ framework with DPU support
- Python 3.10
- DPU bitstream (`dpu.bit`)
- Vitis AI runtime

## Downloading Models

Model files (`.xmodel`) are **not included** in this repository due to their size. You must download them separately from the [Vitis AI Model Zoo](https://github.com/Xilinx/Vitis-AI/tree/master/model_zoo).

### Available Models

| Model | Input Size | Target DPU |
|-------|-----------|------------|
| `mobilenet_v2` | 224×224 | DPUCZDX8G_ISA1_B4096 |
| `mobilenet_v1_1_0_224_tf` | 224×224 | DPUCZDX8G_ISA1_B4096 |
| `resnet50` | 224×224 | DPUCZDX8G_ISA1_B4096 |

### Download Instructions

Download pre-compiled `.xmodel` files for target `DPUCZDX8G_ISA1_B4096` (Zynq UltraScale+ MPSoC / Kria KV260) from the Vitis AI model zoo:

```
https://github.com/Xilinx/Vitis-AI/tree/master/model_zoo/model-list
```

### Installing a Downloaded Model

After downloading, place the files in the correct directory structure:

```
models/
└── your_model/
    ├── meta.json                  # Model metadata (see format below)
    ├── your_model.xmodel          # Compiled DPU model
    └── your_model.prototxt        # Preprocessing parameters
```

**meta.json format:**
```json
{
    "lib": "libvart-dpu-runner.so",
    "filename": "your_model.xmodel",
    "kernel": ["subgraph_name"],
    "target": "DPUCZDX8G_ISA1_B4096"
}
```

**prototxt format:**
```
transform_param {
  mean_value: 104.0
  mean_value: 117.0
  mean_value: 123.0
  scale: 0.00392157
  scale: 0.00392157
  scale: 0.00392157
}
```

> **Note:** The `kernel` name in `meta.json` must match the subgraph name in the `.xmodel` file. You can inspect the xmodel with `xir subgraph <model>.xmodel` on the Kria board.

### Verify Download Integrity

Each model directory contains an `md5sum.txt` file. Verify after downloading:

```bash
cd models/mobilenet_v2
md5sum -c md5sum.txt
```

## Quick Start

### 1. Deploy to Kria

```bash
rsync -avz --exclude=venv --exclude=.git --exclude=__pycache__ ./ ubuntu@192.168.100.8:~/kria-camera-demo/
```

### 2. Run DPU Benchmark with Profiling

```bash
# Apply vaitrace bug fix (first time only)
ssh ubuntu@192.168.100.8 "cd ~/kria-camera-demo && ./patch_vaitrace.sh"

# Run profiling (500 frames)
ssh ubuntu@192.168.100.8 "cd ~/kria-camera-demo && ./run_vaitrace.sh -n 500"

# Download profiling results
scp ubuntu@192.168.100.8:~/kria-camera-demo/*.csv ./
```

### 3. Run Camera Demo

```bash
ssh ubuntu@192.168.100.8 "cd ~/kria-camera-demo && sudo /usr/local/share/pynq-venv/bin/python3 kria-camera-demo.py"
```

## Usage

### DPU Benchmark

The synthetic benchmark is the recommended way to profile DPU performance:

```bash
# Run with vaitrace profiling (recommended)
./run_vaitrace.sh -n 500

# Direct execution (no profiling)
sudo /usr/local/share/pynq-venv/bin/python3 utils/benchmark_dpu.py -n 500
```

**Options:**
```
-n, --num-frames N    Number of frames to process (default: 100)
--warmup N           Warmup iterations (default: 10)
--seed N             Random seed for reproducibility (default: 42)
-m, --model-dir PATH  Model directory (default: models/mobilenet_v2)
-b, --dpu-bit PATH    DPU bitstream file (default: dpu.bit)
-l, --labels PATH     Class labels file (default: words.txt)
```

### Camera Demo

Real-time camera application with visualization:

```bash
# Normal mode with GUI
sudo /usr/local/share/pynq-venv/bin/python3 kria-camera-demo.py

# Headless mode for profiling
./run_vaitrace.sh kria-camera-demo.py

# Custom model
sudo /usr/local/share/pynq-venv/bin/python3 kria-camera-demo.py -m models/resnet50
```

**Options:**
```
-m, --model-dir PATH    Model directory (default: models/mobilenet_v2)
-b, --dpu-bit PATH      DPU bitstream (default: dpu.bit)
-l, --labels PATH       Class labels (default: words.txt)
--headless             Run without GUI (for profiling)
--profile-frames N      Frames in headless mode (default: 100)
```

**Controls:**
- Press `q` to quit

## Profiling with Vaitrace

### Layer-by-Layer DPU Profiling

```bash
# 1. Apply vaitrace patch (fixes KeyError bug)
./patch_vaitrace.sh

# 2. Run profiling
./run_vaitrace.sh -n 500

# 3. Check generated files
ls -lh *.csv
```

**Generated Files:**
- `vart_trace.csv` - Layer-by-layer execution trace
- `vitis_ai_profile.csv` - Detailed DPU profiling data
- `profile_summary.csv` - Performance summary

`run_vaitrace.sh` automatically parses `vart_trace.csv` via `utils/analyze_trace.py` and prints per-layer latency and efficiency statistics when profiling finishes. Save results with `--csv-out`:

```bash
python3 utils/analyze_trace.py --csv-out layer_stats.csv
```

### DDR Traffic Analysis

The `utils/plot_ddr_traffic.py` script visualizes DDR memory bandwidth from profiling results. See `doc/mem_io_metric.md` for an explanation of `Mem IO(MB)` and `Mem Bandwidth(MB/s)` columns in `profile_summary.csv`.

### Vaitrace Configuration

The `run_vaitrace.sh` script uses:
- `--fine_grained` flag for detailed layer profiling
- `--va` flag for VART runtime tracing
- Automatic PYTHONPATH configuration for PYNQ packages

### Known Issue: Vaitrace Bug

Vaitrace has a post-processing bug that causes KeyError crashes. **Apply the patch before profiling:**

```bash
./patch_vaitrace.sh
```

This patches `/usr/bin/xlnx/vaitrace/tracer/function.py` to handle missing function symbols gracefully.

## Project Structure

```
kria-camera-demo/
├── kria-camera-demo.py               # Real-time camera demo
├── run_vaitrace.sh                   # Vaitrace profiling wrapper ⭐
├── patch_vaitrace.sh                 # Fix vaitrace KeyError ⭐
├── utils/                            # Offline analysis and benchmarking tools
│   ├── benchmark_dpu.py              # Synthetic DPU benchmark ⭐
│   ├── analyze_trace.py              # Per-layer trace analysis ⭐
│   ├── plot_ddr_traffic.py           # DDR bandwidth visualization
│   └── sync_timestamps.py            # Timestamp synchronization utility
├── camera_demo/                      # Core package
│   ├── kria_camera_demo.py           # Main camera demo class
│   ├── utils.py                      # Utilities
│   ├── preprocessing.py              # Image preprocessing
│   ├── visualization.py              # Display helpers
│   └── platform_monitor.py           # System monitoring
├── models/                           # Model directories (not in repo — download separately)
│   ├── mobilenet_v2/                 # MobileNet V2
│   ├── mobilenet_v1_1_0_224_tf/      # MobileNet V1 TF
│   └── resnet50/                     # ResNet-50
├── doc/                              # Documentation
│   ├── mem_io_metric.md              # DDR memory metric explanations
│   └── ddrc_port_assignments.md      # DDR controller port reference
├── words.txt                         # ImageNet labels (1000 classes)
├── dpu.bit                           # DPU overlay bitstream
└── requirements.txt                  # Python dependencies
```

⭐ = Essential for DPU profiling

## Installation

### On Development Machine

```bash
git clone <repository-url>
cd kria-camera-demo
```

Model files are not included. See [Downloading Models](#downloading-models).

### On Kria Board

1. Ensure PYNQ environment is installed:
   ```bash
   source /usr/local/share/pynq-venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip3 install -r requirements.txt
   ```

3. Verify DPU bitstream exists:
   ```bash
   ls dpu.bit
   ```

4. Download and place model files in `models/` (see [Downloading Models](#downloading-models)).

## Dependencies

- **pyrealsense2** - Intel RealSense SDK (camera demo only)
- **opencv-python** - Image processing and visualization
- **numpy** - Numerical operations
- **pynq_dpu** - PYNQ DPU overlay support
- **vaitrace_py** - Vitis AI profiling (optional)

## Troubleshooting

### Vaitrace KeyError

**Symptom:**
```
KeyError: '_fun_XXXXXX'
```

**Solution:**
```bash
./patch_vaitrace.sh
```

### DPU Memory Error

**Symptom:**
```
Check failed: fromdata != ((void *) -1)
```

**Solution:** Don't use `XLNX_ENABLE_DUMP=1`. Use vaitrace instead.

### Camera Not Detected

**Solution:**
- Ensure RealSense camera is connected via USB 3.0
- Check camera with: `realsense-viewer`

### Permission Denied

**Solution:** Run with `sudo` - required for DPU hardware access

### Module Not Found

**Solution:**
- Activate PYNQ environment: `source /usr/local/share/pynq-venv/bin/activate`
- Or use full path: `/usr/local/share/pynq-venv/bin/python3`

### Wrong Kernel Name in meta.json

**Symptom:** DPU runner fails to initialize or returns no results.

**Solution:** Inspect the xmodel to find the correct subgraph name:
```bash
xir subgraph models/your_model/your_model.xmodel
```
Update `kernel` in `meta.json` to match.

## Performance Notes

### Expected Performance (MobileNet V2)

- **DPU Inference:** ~6ms (160+ FPS)
- **Overall Throughput:** 25-35 FPS (with camera I/O and preprocessing)
- **Preprocessing:** ~1-2ms
- **Postprocessing:** <1ms

### Optimization Tips

1. Use synthetic benchmark to isolate DPU performance
2. Run with sufficient frames (500+) for accurate statistics
3. Apply vaitrace patch before profiling
4. Check P95/P99 latencies for worst-case analysis

## Network Configuration

Default Kria IP: `192.168.100.8`

Update in `CLAUDE.md` if your board uses a different address.

## Contributing

When adding new models:
1. Download the compiled `.xmodel` for target `DPUCZDX8G_ISA1_B4096`
2. Create a directory under `models/` with the proper structure
3. Add `meta.json`, `.prototxt`, and `md5sum.txt`
4. Test with benchmark first: `./run_vaitrace.sh -m models/new_model -n 100`

## Funding

[![dAIEDGE Project](https://img.shields.io/badge/dAIEDGE-Project-6A5ACD?style=for-the-badge)](https://daiedge.eu/)
[![EU Horizon Europe](https://img.shields.io/badge/Funded%20by-EU%20Horizon%20Europe-003399?style=for-the-badge&logo=europeanunion&logoColor=white)](https://research-and-innovation.ec.europa.eu/funding/funding-opportunities/funding-programmes-and-open-calls/horizon-europe_en)

This work was supported by the **[dAIEDGE Open Call Programme](https://daiedge.eu/)**, funded by the **[European Union's Horizon Europe research and innovation programme](https://research-and-innovation.ec.europa.eu/funding/funding-opportunities/funding-programmes-and-open-calls/horizon-europe_en)**.

## License

This project is licensed under the [Apache License 2.0](LICENSE).

You may use, reproduce, and distribute this work under the terms of the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for the full text.

## References

- [AMD Xilinx Kria KV260](https://www.xilinx.com/products/som/kria/kv260-vision-starter-kit.html)
- [Vitis AI Model Zoo](https://github.com/Xilinx/Vitis-AI/tree/master/model_zoo)
- [Intel RealSense](https://www.intelrealsense.com/)
- [Vitis AI](https://www.xilinx.com/products/design-tools/vitis/vitis-ai.html)
- [PYNQ](http://www.pynq.io/)
