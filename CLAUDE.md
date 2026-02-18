# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains two main applications for AMD Xilinx Kria hardware:
1. **Camera Demo** - Real-time RealSense camera with DPU inference and visualization
2. **Synthetic Benchmark** - DPU performance profiling using random data (no camera required)

Development happens locally with remote deployment to Kria board at `192.168.100.8`.

## Quick Start

```bash
# Deploy to Kria
rsync -avz --exclude=venv --exclude=.git --exclude=__pycache__ ./ ubuntu@192.168.100.8:~/kria-camera-demo/

# Run DPU benchmark with layer-by-layer profiling
ssh ubuntu@192.168.100.8 "cd ~/kria-camera-demo && ./run_vaitrace.sh -n 500"

# Download profiling results
scp ubuntu@192.168.100.8:~/kria-camera-demo/*.csv ./
```

## Applications

### 1. Synthetic Benchmark (utils/benchmark_dpu.py)

**Purpose:** DPU performance profiling without camera dependency.

**Features:**
- Uses random images (no camera required)
- Vaitrace instrumentation with `@vai_tracepoint` decorators
- Comprehensive statistics (mean, std, min, max, P50, P95, P99)
- Warmup phase for accurate measurements
- Reproducible with fixed random seed

**Command-Line Options:**
- `-n, --num-frames N` - Number of frames to process (default: 100)
- `--warmup N` - Warmup iterations (default: 10)
- `--seed N` - Random seed for reproducibility (default: 42)
- `-m, --model-dir PATH` - Model directory (default: models/mobilenet_v2)
- `-b, --dpu-bit PATH` - DPU bitstream file (default: dpu.bit)
- `-l, --labels PATH` - Class labels file (default: words.txt)

**Vaitrace Instrumentation:**
The benchmark includes decorated methods that vaitrace automatically profiles:
- `preprocess_image()` - Image normalization and reshaping
- `run_dpu_inference()` - DPU execution (key operation)
- `postprocess_output()` - Softmax and top-k classification

**Usage:**
```bash
# Direct execution
sudo /usr/local/share/pynq-venv/bin/python3 utils/benchmark_dpu.py -n 500

# With vaitrace profiling (recommended)
./run_vaitrace.sh -n 500
```

### 2. Camera Demo (kria-camera-demo.py)

**Purpose:** Real-time camera inference with visualization.

**Requirements:**
- Physical RealSense camera connected to Kria
- X11 display (DISPLAY=192.168.100.8:0)
- Root permissions

**Command-Line Options:**
- `-m, --model-dir PATH` - Model directory (default: models/mobilenet_v2)
- `-b, --dpu-bit PATH` - DPU bitstream (default: dpu.bit)
- `-l, --labels PATH` - Class labels (default: words.txt)
- `--headless` - Run without GUI (for profiling)
- `--profile-frames N` - Frames in headless mode (default: 100)

**Usage:**
```bash
# Normal mode with GUI
sudo /usr/local/share/pynq-venv/bin/python3 kria-camera-demo.py

# Headless for profiling
./run_vaitrace.sh kria-camera-demo.py
```

## DPU Profiling with Vaitrace

### Standard Workflow

```bash
# 1. Deploy code
rsync -avz --exclude=venv --exclude=.git --exclude=__pycache__ ./ ubuntu@192.168.100.8:~/kria-camera-demo/

# 2. Apply vaitrace bug fix (first time only)
ssh ubuntu@192.168.100.8 "cd ~/kria-camera-demo && ./patch_vaitrace.sh"

# 3. Run profiling (defaults to benchmark_dpu.py)
ssh ubuntu@192.168.100.8 "cd ~/kria-camera-demo && ./run_vaitrace.sh -n 500"

# 4. Download profiling CSV files
scp ubuntu@192.168.100.8:~/kria-camera-demo/*.csv ./
```

### Vaitrace Configuration

The `run_vaitrace.sh` script uses:
- `--fine_grained` - Enables detailed layer-by-layer profiling
- `--va` - Enables VART runtime tracing
- Automatic PYTHONPATH setup for PYNQ packages
- Defaults to `utils/benchmark_dpu.py` with 100 frames
- Automatically runs `utils/analyze_trace.py` after profiling completes

### Vaitrace Bug Fix

**Problem:** Vaitrace has a post-processing bug causing KeyError:
```
KeyError: '_fun_XXXXXX'
File "/usr/bin/xlnx/vaitrace/tracer/function.py", line 416
```

**Solution:** Apply the patch:
```bash
./patch_vaitrace.sh
```

The patch wraps the problematic line in try-except to handle missing symbols gracefully.

**Rollback:**
```bash
sudo cp /usr/bin/xlnx/vaitrace/tracer/function.py.backup_* /usr/bin/xlnx/vaitrace/tracer/function.py
```

### Profiling Output Files

Generated files after vaitrace completes:
- `vart_trace.csv` - Layer-by-layer execution trace with timestamps
- `vitis_ai_profile.csv` - Detailed DPU profiling with utilization metrics
- `profile_summary.csv` - Aggregated performance summary

## Alternative Profiling

### Simple Benchmark (No vaitrace)

```bash
./run_benchmark.sh -n 1000
```

Provides statistics without external profiling overhead.

### Check Available Tools

```bash
./run_xdputil_profile.sh
```

Shows available DPU profiling utilities on the system.

## File Structure

```
kria-camera-demo/
├── kria-camera-demo.py           # Real-time camera demo
├── main.py                       # Placeholder/stub
├── run_vaitrace.sh               # Vaitrace wrapper (--fine_grained --va); runs analyze_trace.py after
├── run_vaitrace_raw.sh           # Vaitrace with error handling
├── run_benchmark.sh              # Simple benchmark runner
├── run_dpu_profile.sh            # DPU profiling runner
├── run_xdputil_profile.sh        # Check profiling tools
├── patch_vaitrace.sh             # Fix vaitrace KeyError bug
├── utils/                        # Offline analysis and benchmarking tools
│   ├── benchmark_dpu.py          # Synthetic DPU benchmark (vaitrace instrumented)
│   └── analyze_trace.py          # Parse vart_trace.csv; per-layer latency/efficiency stats
├── camera_demo/                  # Core package
│   ├── __init__.py               # Package exports
│   ├── kria_camera_demo.py       # KriaCameraDemo class
│   ├── utils.py                  # Utilities (privileges, parsing, labels)
│   ├── preprocessing.py          # Image preprocessing pipeline
│   ├── visualization.py          # Display helpers
│   └── platform_monitor.py       # System stats monitoring
├── models/mobilenet_v2/          # Model files
│   ├── meta.json                 # Model metadata
│   ├── mobilenet_v2.xmodel       # Compiled DPU model
│   └── mobilenet_v2.prototxt     # Normalization parameters
├── words.txt                     # ImageNet class labels
├── dpu.bit                       # DPU overlay bitstream
├── requirements.txt              # Python dependencies
├── README.md                     # User documentation
└── CLAUDE.md                     # This file

## Key Technical Details

### Asyncio Event Loop Handling

DPU initialization requires an asyncio event loop. When running under vaitrace (which uses threads), the code automatically creates one if missing:

```python
try:
    asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
```

Implemented in both `kria_camera_demo.py` and `utils/benchmark_dpu.py`.

### Vaitrace Tracepoint Usage

`vai_tracepoint` is a **decorator**, not a function:

**Correct:**
```python
@vai_tracepoint
def preprocess_image(self, image):
    # code
```

**Incorrect:**
```python
vai_tracepoint("preprocessing", "start")  # This will fail
```

The benchmark includes a fallback dummy decorator when vaitrace is unavailable.

### PYNQ Environment

All applications require the PYNQ virtual environment:
- Python: `/usr/local/share/pynq-venv/bin/python3`
- Root permissions required for DPU hardware access

## Dependencies

- `pyrealsense2` - Intel RealSense SDK (camera demo only)
- `opencv-python` - Image processing and visualization
- `numpy` - Array operations
- `pynq_dpu` - PYNQ DPU overlay support
- `vaitrace_py` - Vitis AI trace profiling (optional)

## Common Issues

### 1. Memory allocation failure with DPU profiling env vars

**Symptom:**
```
Check failed: fromdata != ((void *) -1)
```

**Solution:** Don't use `XLNX_ENABLE_DUMP=1` - use vaitrace instead.

### 2. Vaitrace KeyError

**Symptom:**
```
KeyError: '_fun_XXXXXX'
```

**Solution:** Run `./patch_vaitrace.sh`

### 3. No asyncio event loop

**Symptom:**
```
RuntimeError: There is no current event loop in thread
```

**Solution:** Already fixed in code - creates loop automatically.

## Best Practices

1. **Always patch vaitrace first** - Run `./patch_vaitrace.sh` before profiling
2. **Use synthetic benchmark** - Faster, reproducible, no camera dependency
3. **Use sufficient frames** - At least 100 for warmup, 500+ for statistics
4. **Download CSV files** - Contains layer-by-layer DPU profiling data
5. **Run with sudo** - Required for DPU hardware access

## Example Workflows

### Profile New Model

```bash
# 1. Add model to models/ directory
# 2. Deploy
rsync -avz --exclude=venv --exclude=.git ./ ubuntu@192.168.100.8:~/kria-camera-demo/

# 3. Profile
ssh ubuntu@192.168.100.8 "cd ~/kria-camera-demo && ./run_vaitrace.sh -m models/new_model -n 500"

# 4. Download results
scp ubuntu@192.168.100.8:~/kria-camera-demo/*.csv ./
```

### Quick Performance Check

```bash
ssh ubuntu@192.168.100.8 "cd ~/kria-camera-demo && ./run_benchmark.sh -n 1000"
```

### Debug Camera Issues

```bash
ssh ubuntu@192.168.100.8 "cd ~/kria-camera-demo && sudo /usr/local/share/pynq-venv/bin/python3 kria-camera-demo.py"
```
