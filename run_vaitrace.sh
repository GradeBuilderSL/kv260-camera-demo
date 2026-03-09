#!/bin/bash

############################################################################
# Copyright 2026 GradeBuilder SL                                           #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
############################################################################

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
MEMORY_PROFILE=0

for arg in "$@"; do
    if [[ "$arg" == *.py ]]; then
        SCRIPT_FILE="$arg"
    elif [[ "$arg" == "-b" || "$arg" == "--bypass" ]]; then
        BYPASS=1
    elif [[ "$arg" == "-p" || "$arg" == "--memory-profile" ]]; then
        MEMORY_PROFILE=1
    else
        EXTRA_ARGS="$EXTRA_ARGS $arg"
    fi
done

# Conditionally enable XRT memory/AIE profiling
if [[ "$MEMORY_PROFILE" -eq 1 ]]; then
    export XRT_INI_PATH="$SCRIPT_DIR/xrt_memory_profile.ini"
    echo "Memory profiling enabled (XRT_INI_PATH=$XRT_INI_PATH)"
fi

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

FIGURES_DIR="$SCRIPT_DIR/figures"

if [[ "$BYPASS" -eq 1 ]]; then
    echo "Running (bypass mode, no tracing): $SCRIPT_FILE $EXTRA_ARGS"
    echo ""
    /usr/local/share/pynq-venv/bin/python $SCRIPT_FILE $EXTRA_ARGS 2>&1 | tee "$SCRIPT_DIR/benchmark_log.txt"
else
    echo "Running vaitrace on: $SCRIPT_FILE $EXTRA_ARGS"
    echo ""
    /usr/local/share/pynq-venv/bin/python -m vaitrace_py --fine_grained --va $SCRIPT_FILE $EXTRA_ARGS 2>&1 | tee "$SCRIPT_DIR/benchmark_log.txt"

    # Analyze the generated trace (optionally include memory profile report)
    if [[ -f "$SCRIPT_DIR/xrt.run_summary" ]]; then
        echo ""
        echo "--- Trace Analysis ---"
        ANALYZE_ARGS="$SCRIPT_DIR/xrt.run_summary --plot --plot-out $FIGURES_DIR"
        if [[ "$MEMORY_PROFILE" -eq 1 ]]; then
            ANALYZE_ARGS="$ANALYZE_ARGS --memory-profile"
        fi
        /usr/local/share/pynq-venv/bin/python "$SCRIPT_DIR/utils/analyze_trace.py" $ANALYZE_ARGS
    else
        echo "Warning: xrt.run_summary not found, skipping trace analysis"
    fi

    # DDR bandwidth timeline figure
    if [[ -f "$SCRIPT_DIR/vitis_ai_profile.csv" ]]; then
        echo ""
        echo "--- DDR Traffic Plot ---"
        /usr/local/share/pynq-venv/bin/python "$SCRIPT_DIR/utils/plot_ddr_traffic.py" \
            "$SCRIPT_DIR/vitis_ai_profile.csv" \
            --out-dir "$FIGURES_DIR" \
            || echo "Warning: DDR traffic plot failed"
    fi
fi
