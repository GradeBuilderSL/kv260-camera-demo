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
Plot DDR traffic timeline for a vaitrace benchmark run.

Two-panel figure:
  Top    — stacked read (DPU ports S1+S2 / CPU ports S3+S4) + total write overlay
  Bottom — total read vs total write with global mean lines

Phase markers (init / DPU inference / finalise) are overlaid when
benchmark_log.txt and vart_trace.csv are present in the same directory.

Timestamp alignment:
  vitis_ai_profile.csv  raw CLOCK_MONOTONIC ms (steady_clock).
                        APM starts before the Python process → DDR_min = t=0.
  benchmark_log.txt     Python time.monotonic()*1000 ms (same clock).
                        First [ts] entry = T0_log = Python process start.
  vart_trace.csv        ms elapsed since T0_log (process-relative).

  T0_offset   = (T0_log  - DDR_min) / 1000   [Python start on DDR axis]
  T_DPU_START = T0_offset + vart_min / 1000  [first DPU kernel]
  T_DPU_END   = T0_offset + vart_max / 1000  [last  DPU kernel]

Usage:
    python plot_ddr_traffic.py [vitis_ai_profile.csv] [--out-dir DIR]
"""

import argparse
import re
import sys
from io import StringIO
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        'csv', nargs='?', default='vitis_ai_profile.csv',
        help='Path to vitis_ai_profile.csv (default: vitis_ai_profile.csv)',
    )
    p.add_argument(
        '--out-dir', type=Path, default=None,
        help='Directory for output figures (default: figures/ next to CSV)',
    )
    return p.parse_args()


# ── data loaders ──────────────────────────────────────────────────────────────

def load_ddr(csv_path: Path):
    """Return sorted DDR DataFrame with 't' column (seconds from DDR_min) and DDR_min."""
    with open(csv_path) as fh:
        lines = fh.readlines()
    idx = next(i for i, l in enumerate(lines) if l.strip() == 'DDR Bandwidth')
    raw = pd.read_csv(StringIO(''.join(lines[idx + 1:])))
    raw = raw.loc[:, (raw != 0).any(axis=0)]
    raw = raw.sort_values('timestamp').reset_index(drop=True)
    ddr_min = raw['timestamp'].iloc[0]
    raw['t'] = (raw['timestamp'] - ddr_min) / 1000.0
    return raw, ddr_min


def try_load_phase_markers(data_dir: Path, ddr_min: float):
    """
    Return (T0_offset, T_DPU_START, T_DPU_END) in seconds on the DDR axis,
    or None if benchmark_log.txt or vart_trace.csv are missing / unparseable.
    """
    log_path  = data_dir / 'benchmark_log.txt'
    vart_path = data_dir / 'vart_trace.csv'
    if not log_path.exists() or not vart_path.exists():
        return None

    # First [timestamp] entry in benchmark_log → T0_log (boot-relative ms)
    T0_LOG = None
    with open(log_path) as fh:
        for line in fh:
            m = re.match(r'^\[(\d+\.\d+)\]', line.strip())
            if m:
                T0_LOG = float(m.group(1))
                break
    if T0_LOG is None:
        return None

    # vart_trace EVENTS timestamp range (process-relative ms)
    vts = []
    in_events = False
    with open(vart_path) as fh:
        for line in fh:
            if line.strip() == 'EVENTS':
                in_events = True
                continue
            if in_events and line.strip():
                parts = line.strip().split(',')
                try:
                    vts.append(float(parts[2]))
                except (ValueError, IndexError):
                    pass
    if not vts:
        return None

    T0_OFFSET   = (T0_LOG - ddr_min) / 1000.0
    T_DPU_START = T0_OFFSET + min(vts) / 1000.0
    T_DPU_END   = T0_OFFSET + max(vts) / 1000.0
    return T0_OFFSET, T_DPU_START, T_DPU_END


# ── figure ────────────────────────────────────────────────────────────────────

def plot(raw, markers, out_dir: Path) -> None:
    # Port grouping (Zynq UltraScale+ MPSoC DDRC):
    #   S1 → APU (Cortex-A53) via CCI-400 coherent interconnect  ← CPU
    #   S2 → LPD masters (DMA, RPU) via CCI-400                  ← CPU/DMA
    #   S3 → HP0/HP1 FPD high-performance PL ports               ← DPU (M_AXI_HP0)
    #   S4 → HP2/HP3 FPD high-performance PL ports               ← DPU (M_AXI_HP2)
    # Reference: UG1085 Zynq UltraScale+ TRM, PG338 DPUCZDX8G Product Guide
    read_cols  = [c for c in raw.columns if c.endswith('_Read')]
    write_cols = [c for c in raw.columns if c.endswith('_Write')]
    dpu_r = [c for c in read_cols if c in ('DDRC_PORT_S3_Read', 'DDRC_PORT_S4_Read')]
    cpu_r = [c for c in read_cols if c in ('DDRC_PORT_S1_Read', 'DDRC_PORT_S2_Read')]

    raw['dpu_read']  = raw[dpu_r].sum(axis=1) if dpu_r  else 0
    raw['cpu_read']  = raw[cpu_r].sum(axis=1) if cpu_r  else 0
    raw['tot_read']  = raw['dpu_read'] + raw['cpu_read']
    raw['tot_write'] = raw[write_cols].sum(axis=1) if write_cols else 0

    # Rolling-mean smoothing (window = 5 samples ≈ 50 ms)
    W  = 5
    df = raw.copy()
    for col in ('dpu_read', 'cpu_read', 'tot_read', 'tot_write'):
        df[col] = df[col].rolling(W, center=True, min_periods=1).mean()

    t          = df['t'].values
    T_MAX      = t[-1]
    mean_read  = raw['tot_read'].mean()
    mean_write = raw['tot_write'].mean()
    YMAX       = raw['tot_read'].max() * 1.12

    ph_alpha     = 0.16
    phase_colors = {'init': '#b8b8b8', 'dpu': '#a8d4f0', 'post': '#ffd8a8'}

    def shade_phases(ax):
        if markers is None:
            return
        T0, T_DPU_S, T_DPU_E = markers
        ax.axvspan(0,        T_DPU_S, color=phase_colors['init'], alpha=ph_alpha, zorder=0)
        ax.axvspan(T_DPU_S, T_DPU_E, color=phase_colors['dpu'],  alpha=ph_alpha, zorder=0)
        ax.axvspan(T_DPU_E, T_MAX,   color=phase_colors['post'], alpha=ph_alpha, zorder=0)
        for xb in (T_DPU_S, T_DPU_E):
            ax.axvline(xb, color='#555', lw=0.8, ls=':', zorder=5)
        ax.axvline(T0, color='#444', lw=0.7, ls='--', alpha=0.6, zorder=5)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                             gridspec_kw={'hspace': 0.10})

    # ── top panel: stacked read + write overlay ────────────────────────────────
    ax0 = axes[0]
    shade_phases(ax0)
    ax0.stackplot(
        t, df['dpu_read'].values, df['cpu_read'].values,
        colors=['#2166ac', '#74add1'], alpha=0.82,
        labels=['Read – DPU/PL ports (S3+S4)', 'Read – CPU/DMA ports (S1+S2)'],
    )
    ax0.plot(t, df['tot_write'].values,
             color='#d73027', lw=1.1, ls='--', label='Write (all ports)')

    if markers:
        T0, T_DPU_S, T_DPU_E = markers
        label_y = YMAX * 0.07
        for mid, lbl in (
            ((0 + T_DPU_S) / 2,       'vaitrace init\n& overlay load'),
            ((T_DPU_S + T_DPU_E) / 2, 'DPU inference'),
            ((T_DPU_E + T_MAX) / 2,   'APM\nfinalise'),
        ):
            ax0.text(
                mid, label_y, lbl, ha='center', va='bottom', fontsize=6.5,
                color='#444',
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.75),
            )
        ax0.axvline(T0, color='#444', lw=0.7, ls='--', alpha=0.6, zorder=5)
        ax0.text(T0 + (T_MAX * 0.006), YMAX * 0.91, 'Python\nstart',
                 ha='left', va='top', fontsize=6, color='#444')

    ax0.set_ylabel('DDR bandwidth (MB/s)', fontsize=9)
    ax0.set_ylim(0, YMAX)
    ax0.tick_params(labelsize=7.5)
    ax0.legend(loc='upper left', fontsize=7, framealpha=0.85, ncol=1, handlelength=1.6)

    # ── bottom panel: total read vs write with means ───────────────────────────
    ax1 = axes[1]
    shade_phases(ax1)
    ax1.fill_between(t, df['tot_read'].values,  color='#2166ac', alpha=0.22)
    ax1.fill_between(t, df['tot_write'].values, color='#d73027', alpha=0.18)
    ax1.plot(t, df['tot_read'].values,  color='#2166ac', lw=1.2, label='Total Read')
    ax1.plot(t, df['tot_write'].values, color='#d73027', lw=1.2, ls='--', label='Total Write')
    ax1.axhline(mean_read,  color='#2166ac', lw=0.8, ls='--', alpha=0.65)
    ax1.axhline(mean_write, color='#d73027', lw=0.8, ls='--', alpha=0.65)
    ax1.text(T_MAX * 0.97, mean_read  + YMAX * 0.02,
             f'$\\bar{{x}}$={mean_read:.0f}',
             ha='right', va='bottom', fontsize=6.5, color='#2166ac')
    ax1.text(T_MAX * 0.97, mean_write + YMAX * 0.02,
             f'$\\bar{{x}}$={mean_write:.0f}',
             ha='right', va='bottom', fontsize=6.5, color='#d73027')

    if markers:
        for xb in (markers[1], markers[2]):
            ax1.axvline(xb, color='#555', lw=0.8, ls=':', zorder=5)
        ax1.axvline(markers[0], color='#444', lw=0.7, ls='--', alpha=0.6, zorder=5)

    ax1.set_ylabel('MB/s', fontsize=9)
    ax1.set_xlabel('Time (s) from APM start', fontsize=9)
    ax1.set_ylim(0, YMAX)
    ax1.set_xlim(0, T_MAX)
    ax1.tick_params(labelsize=7.5)
    ax1.legend(loc='upper left', fontsize=7, framealpha=0.85, handlelength=1.6)

    title = 'DDR Bandwidth Timeline'
    if markers:
        title += '  ·  phase markers from vart_trace + benchmark_log'
    fig.suptitle(title, fontsize=10, y=1.01)

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        out = out_dir / f'ddr_traffic.{ext}'
        fig.savefig(out, bbox_inches='tight', dpi=150)
        print(f'Saved {out}')

    if markers:
        T0, T_DPU_S, T_DPU_E = markers
        print(f'Phase boundaries: T0_offset={T0:.3f}s  '
              f'T_DPU_START={T_DPU_S:.3f}s  T_DPU_END={T_DPU_E:.3f}s  T_MAX={T_MAX:.3f}s')
    print(f'Means: read={mean_read:.1f} MB/s  write={mean_write:.1f} MB/s  '
          f'|  Samples: {len(raw)}')

    plt.close(fig)


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    args     = parse_args()
    csv_path = Path(args.csv)

    if not csv_path.exists():
        print(f'Error: {csv_path} not found', file=sys.stderr)
        sys.exit(1)

    out_dir = args.out_dir or csv_path.parent / 'figures'

    print(f'Parsing: {csv_path}')
    raw, DDR_MIN = load_ddr(csv_path)

    markers = try_load_phase_markers(csv_path.parent, DDR_MIN)
    if markers:
        print(f'Phase markers loaded from benchmark_log.txt + vart_trace.csv')
    else:
        print('Phase markers not available (benchmark_log.txt or vart_trace.csv missing)')

    plot(raw, markers, out_dir)


if __name__ == '__main__':
    main()
