#!/bin/bash
# Wrapper script for running vaitrace_py with PYNQ virtual environment support
# This script ensures that vaitrace can access both system packages and PYNQ packages
# By default, runs the synthetic benchmark (benchmark_dpu.py)

set -e

# Set PYTHONPATH to include PYNQ virtual environment packages and project root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR":/usr/local/share/pynq-venv/lib/python3.10/site-packages:/usr/local/lib/python3.10/dist-packages

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "This script requires root privileges. Running with sudo..."
    exec sudo -E PYTHONPATH="$PYTHONPATH" "$0" "$@"
fi

# Parse arguments to find Python script
SCRIPT_FILE=""
EXTRA_ARGS=""
BYPASS=0

for arg in "$@"; do
    if [[ "$arg" == *.py ]]; then
        SCRIPT_FILE="$arg"
    elif [[ "$arg" == "-b" || "$arg" == "--bypass" ]]; then
        BYPASS=1
    else
        EXTRA_ARGS="$EXTRA_ARGS $arg"
    fi
done

# Default to benchmark_dpu.py if no script specified
if [[ -z "$SCRIPT_FILE" ]]; then
    SCRIPT_FILE="utils/benchmark_dpu.py"
    echo "No script specified, using synthetic benchmark: $SCRIPT_FILE"
fi

# Script-specific configuration
if [[ "$SCRIPT_FILE" == "utils/benchmark_dpu.py" ]]; then
    # Ensure benchmark has reasonable defaults
    if [[ "$EXTRA_ARGS" != *"-n"* ]] && [[ "$EXTRA_ARGS" != *"--num-frames"* ]]; then
        echo "Using default 100 frames for benchmark"
        EXTRA_ARGS="$EXTRA_ARGS -n 100"
    fi
fi

if [[ "$BYPASS" -eq 1 ]]; then
    echo "Running (bypass mode, no tracing): $SCRIPT_FILE $EXTRA_ARGS"
    echo ""
    /usr/local/share/pynq-venv/bin/python $SCRIPT_FILE $EXTRA_ARGS
else
    echo "Running vaitrace on: $SCRIPT_FILE $EXTRA_ARGS"
    echo ""
    /usr/local/share/pynq-venv/bin/python -m vaitrace_py --fine_grained --va $SCRIPT_FILE $EXTRA_ARGS

    # Analyze the generated trace
    if [[ -f "$SCRIPT_DIR/xrt.run_summary" ]]; then
        echo ""
        echo "--- Trace Analysis ---"
        python3 "$SCRIPT_DIR/utils/analyze_trace.py" "$SCRIPT_DIR/xrt.run_summary"
    else
        echo "Warning: xrt.run_summary not found, skipping trace analysis"
    fi
fi
