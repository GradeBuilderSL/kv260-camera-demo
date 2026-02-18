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

# Quick benchmark without profiling
./run_benchmark.sh -n 1000

# Direct execution
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
sudo /usr/local/share/pynq-venv/bin/python3 kria-camera-demo.py -m models/custom_model
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

The benchmark includes vaitrace instrumentation for detailed profiling:

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

`run_vaitrace.sh` automatically parses `vart_trace.csv` via `utils/analyze_trace.py` and prints per-layer latency and efficiency statistics when profiling finishes. Use `--csv-out` to save results:

```bash
python3 utils/analyze_trace.py --csv-out layer_stats.csv
```

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
├── kria-camera-demo.py           # Real-time camera demo
├── run_vaitrace.sh               # Vaitrace profiling wrapper ⭐
├── run_benchmark.sh              # Simple benchmark runner
├── run_dpu_profile.sh            # DPU profiling runner
├── run_xdputil_profile.sh        # Check profiling tools
├── patch_vaitrace.sh             # Fix vaitrace KeyError ⭐
├── utils/                        # Offline tools
│   ├── benchmark_dpu.py          # Synthetic DPU benchmark ⭐
│   └── analyze_trace.py          # Per-layer trace analysis ⭐
├── camera_demo/                  # Core package
│   ├── kria_camera_demo.py       # Main camera demo class
│   ├── utils.py                  # Utilities
│   ├── preprocessing.py          # Image preprocessing
│   ├── visualization.py          # Display helpers
│   └── platform_monitor.py       # System monitoring
├── models/mobilenet_v2/          # Model files
│   ├── meta.json                 # Model metadata
│   ├── mobilenet_v2.xmodel       # Compiled DPU model
│   └── mobilenet_v2.prototxt     # Normalization params
├── words.txt                     # ImageNet labels (1000 classes)
├── dpu.bit                       # DPU overlay bitstream
└── requirements.txt              # Python dependencies
```

⭐ = Essential for DPU profiling

## Model Directory Structure

Each model directory must contain:

```
models/your_model/
├── meta.json              # Model metadata
├── your_model.xmodel      # Compiled DPU model
└── your_model.prototxt    # Preprocessing config
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

## Installation

### On Development Machine

```bash
git clone <repository-url>
cd kria-camera-demo
```

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

Update in `.vscode/tasks.json` and `CLAUDE.md` if your board uses a different address.

## Contributing

When adding new models:
1. Place in `models/` directory with proper structure
2. Include meta.json and prototxt
3. Test with benchmark first: `./run_benchmark.sh -m models/new_model -n 100`
4. Profile with vaitrace: `./run_vaitrace.sh -m models/new_model -n 500`

## License

[Add your license here]

## References

- [AMD Xilinx Kria KV260](https://www.xilinx.com/products/som/kria/kv260-vision-starter-kit.html)
- [Intel RealSense](https://www.intelrealsense.com/)
- [Vitis AI](https://www.xilinx.com/products/design-tools/vitis/vitis-ai.html)
- [PYNQ](http://www.pynq.io/)
