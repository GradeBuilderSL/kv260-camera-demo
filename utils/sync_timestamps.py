#!/usr/bin/env python3

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

"""
Timestamp synchronisation analysis for MobileNet V2 vaitrace benchmark.

Three files, three clocks — all from the same run:

  vitis_ai_profile.csv   raw CLOCK_MONOTONIC ms (C++ steady_clock).
                         APM hardware counter starts BEFORE the Python
                         process begins logging → DDR_min is t=0 for
                         the whole run.

  benchmark_log.txt      Python time.monotonic()*1000 ms (same clock).
                         First [ts] entry = T0_log = Python process start.

  vart_trace.csv         ms elapsed since T0_log (process-relative).
                         Derived from ftrace boot clock; vaitrace's
                         do_timeline_sync() sets zero = T0_log.

Alignment:
    x-axis = (DDR_ts - DDR_min) / 1000          [seconds from APM start]

    T0_offset   = (T0_log  - DDR_min) / 1000    [Python start on DDR axis]
    T_DPU_START = T0_offset + vart_min / 1000   [first DPU kernel]
    T_DPU_END   = T0_offset + vart_max / 1000   [last  DPU kernel]

Cross-check: DDR_max ≈ T0_offset + ftrace_stop_s  (APM and ftrace stop together)

Usage:
    python sync_timestamps.py            # uses default paths below
    python sync_timestamps.py --log /path/to/folder
"""

import argparse
import re
import sys
from io import StringIO
from pathlib import Path

import pandas as pd


# ── default paths ──────────────────────────────────────────────────────────────
DEFAULT_DIR = Path('/home/ivan/projects/logs/mobilenet_v2')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--log', type=Path, default=DEFAULT_DIR,
                   help='Directory containing vitis_ai_profile.csv, '
                        'benchmark_log.txt, and vart_trace.csv')
    return p.parse_args()


def load_ddr(csv_path: Path):
    """Return sorted DDR DataFrame with a 't' column (seconds from DDR_min)."""
    with open(csv_path) as fh:
        lines = fh.readlines()
    ddr_start = next(i for i, l in enumerate(lines) if l.strip() == 'DDR Bandwidth')
    raw = pd.read_csv(StringIO(''.join(lines[ddr_start + 1:])))
    raw = raw.loc[:, (raw != 0).any(axis=0)]
    raw = raw.sort_values('timestamp').reset_index(drop=True)
    ddr_min = raw['timestamp'].iloc[0]
    raw['t'] = (raw['timestamp'] - ddr_min) / 1000.0
    return raw, ddr_min


def load_t0_log(log_path: Path) -> float:
    """Return T0_log: first boot-relative ms timestamp from benchmark_log.txt."""
    with open(log_path) as fh:
        for line in fh:
            m = re.match(r'^\[(\d+\.\d+)\]', line.strip())
            if m:
                return float(m.group(1))
    raise ValueError(f'No [timestamp] entries found in {log_path}')


def load_vart_range(vart_path: Path):
    """Return (vart_min_ms, vart_max_ms) from vart_trace.csv EVENTS section."""
    timestamps = []
    in_events = False
    with open(vart_path) as fh:
        for line in fh:
            if line.strip() == 'EVENTS':
                in_events = True
                continue
            if in_events and line.strip():
                parts = line.strip().split(',')
                try:
                    timestamps.append(float(parts[2]))
                except (ValueError, IndexError):
                    pass
    if not timestamps:
        raise ValueError(f'No event timestamps found in {vart_path}')
    return min(timestamps), max(timestamps)


def main():
    args = parse_args()
    d = args.log

    csv_path  = d / 'vitis_ai_profile.csv'
    log_path  = d / 'benchmark_log.txt'
    vart_path = d / 'vart_trace.csv'

    for p in (csv_path, log_path, vart_path):
        if not p.exists():
            print(f'ERROR: file not found: {p}', file=sys.stderr)
            sys.exit(1)

    # ── load data ──────────────────────────────────────────────────────────────
    raw, DDR_MIN    = load_ddr(csv_path)
    T0_LOG          = load_t0_log(log_path)
    vart_min, vart_max = load_vart_range(vart_path)

    # ── derived boundaries ─────────────────────────────────────────────────────
    T0_OFFSET   = (T0_LOG - DDR_MIN) / 1000.0
    T_DPU_START = T0_OFFSET + vart_min / 1000.0
    T_DPU_END   = T0_OFFSET + vart_max / 1000.0
    T_MAX       = raw['t'].iloc[-1]

    # ── DDR channel sums ───────────────────────────────────────────────────────
    # Port grouping (Zynq UltraScale+ MPSoC DDRC):
    #   S1/S2 → APU + LPD masters via CCI-400  (CPU/DMA)
    #   S3/S4 → HP0-HP3 FPD PL ports           (DPU via M_AXI_HP0/HP2)
    # Reference: UG1085 Zynq UltraScale+ TRM, PG338 DPUCZDX8G Product Guide
    raw['dpu_read']  = raw['DDRC_PORT_S3_Read']  + raw['DDRC_PORT_S4_Read']
    raw['cpu_read']  = raw['DDRC_PORT_S1_Read']  + raw['DDRC_PORT_S2_Read']
    raw['tot_read']  = raw['dpu_read'] + raw['cpu_read']
    raw['tot_write'] = (raw['DDRC_PORT_S1_Write'] + raw['DDRC_PORT_S2_Write'] +
                        raw['DDRC_PORT_S3_Write'] + raw['DDRC_PORT_S4_Write'])

    # ── report ─────────────────────────────────────────────────────────────────
    print('=== Clock values (raw) ===')
    print(f'  DDR_min  (APM start,  boot-rel ms) : {DDR_MIN:.3f}   '
          f'= {DDR_MIN/1000/3600:.4f} h uptime')
    print(f'  T0_log   (Python start, boot-rel ms): {T0_LOG:.3f}   '
          f'= {T0_LOG/1000/3600:.4f} h uptime')
    print(f'  vart_min (first DPU kernel, proc-rel ms): {vart_min:.3f}')
    print(f'  vart_max (last  DPU kernel, proc-rel ms): {vart_max:.3f}')
    print()

    print('=== Aligned timeline (seconds from APM start = DDR_min) ===')
    print(f'  T0_offset   = {T0_OFFSET:.3f} s  (Python process start)')
    print(f'  T_DPU_START = {T_DPU_START:.3f} s  (first DPU kernel, vart_trace)')
    print(f'  T_DPU_END   = {T_DPU_END:.3f} s  (last  DPU kernel, vart_trace)')
    print(f'  T_MAX       = {T_MAX:.3f} s  (last DDR sample)')
    print()

    print('=== Phase boundaries ===')
    print(f'  Phase 1 — init & overlay load : 0.000 .. {T_DPU_START:.3f} s '
          f'({T_DPU_START:.3f} s)')
    print(f'  Phase 2 — DPU inference        : {T_DPU_START:.3f} .. {T_DPU_END:.3f} s '
          f'({T_DPU_END - T_DPU_START:.3f} s,  {vart_max - vart_min:.0f} ms)')
    print(f'  Phase 3 — APM finalise         : {T_DPU_END:.3f} .. {T_MAX:.3f} s '
          f'({T_MAX - T_DPU_END:.3f} s)')
    print()

    print('=== DDR statistics ===')
    print(f'  Samples           : {len(raw)}')
    print(f'  Mean read  (total): {raw["tot_read"].mean():.1f} MB/s')
    print(f'  Mean write (total): {raw["tot_write"].mean():.1f} MB/s')
    print(f'  Peak read  (total): {raw["tot_read"].max():.1f} MB/s')
    print(f'  Peak write (total): {raw["tot_write"].max():.1f} MB/s')
    print(f'  Mean read  DPU (S3+S4): {raw["dpu_read"].mean():.1f} MB/s')
    print(f'  Mean read  CPU (S1+S2): {raw["cpu_read"].mean():.1f} MB/s')
    print()

    print('=== Cross-check ===')
    # APM should stop at ~ftrace stop from benchmark_log
    # ftrace stop is logged; we can't read it here without parsing more of the log,
    # but we know empirically it's T0_OFFSET + ~6.998 s
    expected_ftrace_stop = T0_OFFSET + 6.998
    print(f'  T_MAX (last DDR sample)  = {T_MAX:.3f} s')
    print(f'  Expected ftrace stop     ≈ {expected_ftrace_stop:.3f} s  '
          f'(T0_offset + 6.998 s from benchmark_log)')
    print(f'  Difference               = {abs(T_MAX - expected_ftrace_stop)*1000:.1f} ms '
          f'(should be < 20 ms)')

    return {
        'DDR_MIN': DDR_MIN,
        'T0_LOG': T0_LOG,
        'T0_OFFSET': T0_OFFSET,
        'T_DPU_START': T_DPU_START,
        'T_DPU_END': T_DPU_END,
        'T_MAX': T_MAX,
    }


if __name__ == '__main__':
    main()
