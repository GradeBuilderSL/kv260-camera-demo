#!/usr/bin/env python3
"""
Analyze vart_trace.csv (VP_TRACE) from xrt.run_summary.

Parses the MAPPING and EVENTS sections to compute per-layer
averaged latency (ms) and efficiency (GOP/s) with mean ± std dev.

Column layout in EVENTS (VTF format):
  0: EventID     - unique event identifier
  1: StartID     - 0 → start event; non-zero → end event paired with StartID
  2: Timestamp   - ms
  3: BucketID    - DPU core ID (coreId + 1)
  4: Type        - "VART_RUNNER"
  5: KernelID    - index into MAPPING section (subgraph name)
  6: PID/ThreadID
  7: BatchSize
  8: Workload    - computation in GOPS
  9: Efficiency  - DPU throughput in GOP/s
 10: OpNum       - number of ops in subgraph
 11: InputTensorMID
 12: OutputTensorMID
"""

import json
import os
import sys
import csv
import statistics
from pathlib import Path
from collections import defaultdict


# ---------------------------------------------------------------------------
# Memory profile helpers — profile_summary.csv and vitis_ai_profile.csv
# ---------------------------------------------------------------------------

def _parse_sections(path: Path) -> dict:
    """
    Parse a Vitis AI multi-section CSV file.

    The format uses bare section-title lines (not valid CSV) followed by a
    CSV header row and data rows.  Sections are separated by blank lines.

    Returns {section_title: (header_list, [row_dict, ...])}
    """
    sections = {}
    current_title = None
    current_header = None
    current_rows = []

    with open(path, newline="") as f:
        for raw in f:
            line = raw.rstrip("\n")

            if not line.strip():
                # blank line → flush current section
                if current_title and current_header is not None:
                    sections[current_title] = (current_header, current_rows)
                current_title = None
                current_header = None
                current_rows = []
                continue

            parts = [p.strip() for p in line.split(",")]
            # Remove trailing empty token from lines that end with ","
            if parts and parts[-1] == "":
                parts = parts[:-1]

            if current_header is None:
                # Is this a data-looking line (starts with a number)?
                try:
                    float(parts[0])
                    # Looks like data but we have no header yet — skip
                    continue
                except ValueError:
                    pass

                if current_title is None:
                    # First non-blank non-numeric line → section title
                    # (Could be "Profile Summary", "DPU Summary", etc.)
                    # Skip metadata lines that contain ":" (key: value)
                    if len(parts) == 1 or (len(parts) >= 2 and ":" not in parts[0]):
                        current_title = parts[0]
                    # else: skip metadata key:value lines
                else:
                    # We have a title; this is the CSV header row
                    current_header = parts
            else:
                # Data row
                if len(parts) == 0:
                    continue
                row = dict(zip(current_header, parts))
                current_rows.append(row)

    # Flush last section
    if current_title and current_header is not None:
        sections[current_title] = (current_header, current_rows)

    return sections


def _stats(vals: list) -> dict:
    if not vals:
        return {}
    return {
        "n":    len(vals),
        "mean": statistics.mean(vals),
        "std":  statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "min":  min(vals),
        "max":  max(vals),
        "sum":  sum(vals),
    }


def print_dpu_memory_stats(path: Path) -> None:
    """
    Parse profile_summary.csv and display per-kernel Mem IO and Mem Bandwidth.

    Only DPU kernels (rows that have CU Full Name containing "DPUCZDX8G")
    carry Mem IO(MB) and Mem Bandwidth(MB/s); CPU rows are excluded.
    """
    if not path.exists():
        print(f"  {path.name} not found — skipping DPU memory stats.")
        return

    sections = _parse_sections(path)
    header, rows = sections.get("DPU Summary", (None, []))
    if not rows:
        print("  No 'DPU Summary' section found in profile_summary.csv.")
        return

    mem_io_col  = next((h for h in (header or []) if "mem io"  in h.lower()), None)
    mem_bw_col  = next((h for h in (header or []) if "mem bandwidth" in h.lower()), None)
    name_col    = next((h for h in (header or []) if "kernel name" in h.lower()), None)
    cu_col      = next((h for h in (header or []) if "cu full name" in h.lower()), None)
    runs_col    = next((h for h in (header or []) if "number of runs" in h.lower()), None)
    avg_bw_col  = next((h for h in (header or []) if "mem bandwidth" in h.lower()), None)

    if not mem_io_col or not mem_bw_col:
        print("  Mem IO / Mem Bandwidth columns not found in profile_summary.csv.")
        return

    # Keep only DPU rows (CU name contains "DPUCZDX8G")
    dpu_rows = [
        r for r in rows
        if cu_col and "DPUCZDX8G" in r.get(cu_col, "")
    ]
    if not dpu_rows:
        print("  No DPU kernel rows found in profile_summary.csv.")
        return

    name_w = max(len(r.get(name_col, "")) for r in dpu_rows) + 2
    print(f"\n  {'Kernel':<{name_w}} {'Runs':>5}  {'Mem IO(MB)':>12}  {'Mem BW(MB/s)':>13}")
    print(f"  {'-' * (name_w + 5 + 14 + 15)}")

    mem_io_vals, mem_bw_vals = [], []
    for r in dpu_rows:
        name = r.get(name_col, "?")
        runs = r.get(runs_col, "?")
        try:
            io_val = float(r.get(mem_io_col, ""))
            bw_val = float(r.get(mem_bw_col, ""))
            mem_io_vals.append(io_val)
            mem_bw_vals.append(bw_val)
            print(f"  {name:<{name_w}} {runs:>5}  {io_val:>12.3f}  {bw_val:>13.3f}")
        except ValueError:
            print(f"  {name:<{name_w}} {runs:>5}  {'N/A':>12}  {'N/A':>13}")

    if mem_io_vals:
        print()
        print(f"  Total Mem IO : {sum(mem_io_vals):.3f} MB  (per-kernel sum)")
        s = _stats(mem_bw_vals)
        print(
            f"  Mem BW       : mean={s['mean']:.1f}  min={s['min']:.1f}  "
            f"max={s['max']:.1f}  MB/s  (across {s['n']} DPU kernels)"
        )


def print_ddr_bandwidth_stats(path: Path) -> None:
    """
    Parse vitis_ai_profile.csv and display DDR bandwidth statistics.

    The "DDR Bandwidth" section contains a time-series of per-port read/write
    samples in MB/s.  Only ports with at least one non-zero sample are shown.
    Aggregate read/write totals are computed across active ports.
    """
    if not path.exists():
        print(f"  {path.name} not found — skipping DDR bandwidth stats.")
        return

    sections = _parse_sections(path)
    header, rows = sections.get("DDR Bandwidth", (None, []))
    if not rows:
        print("  No 'DDR Bandwidth' section found in vitis_ai_profile.csv.")
        return

    # Columns: timestamp, DDRC_PORT_S1_Read, DDRC_PORT_S1_Write, ...
    port_cols = [h for h in (header or []) if h.startswith("DDRC_PORT")]
    if not port_cols:
        print("  No DDRC_PORT columns found.")
        return

    # Collect time-series per port column
    series: dict = {col: [] for col in port_cols}
    for row in rows:
        for col in port_cols:
            try:
                series[col].append(float(row.get(col, 0)))
            except ValueError:
                series[col].append(0.0)

    # Identify active ports (at least one non-zero sample)
    active = [c for c in port_cols if any(v > 0 for v in series[c])]
    if not active:
        print("  All DDR ports show zero bandwidth — no DPU activity captured.")
        return

    # Group into port pairs (S1, S2, ...) → {port: {Read: [], Write: []}}
    port_groups: dict = {}
    for col in active:
        # col = "DDRC_PORT_S1_Read"  →  port="DDRC_PORT_S1", dir="Read"
        if "_Read" in col:
            port_name, direction = col.rsplit("_Read", 1)[0], "Read"
        elif "_Write" in col:
            port_name, direction = col.rsplit("_Write", 1)[0], "Write"
        else:
            continue
        port_groups.setdefault(port_name, {})
        port_groups[port_name][direction] = series[col]

    name_w = max(len(p) for p in port_groups) + 2
    hdr = (
        f"  {'Port':<{name_w}}"
        f"  {'Read mean':>10}  {'Read max':>9}"
        f"  {'Write mean':>11}  {'Write max':>10}  (MB/s)"
    )
    print(hdr)
    print(f"  {'-' * (len(hdr) - 2)}")

    total_read_ts  = [0.0] * len(rows)
    total_write_ts = [0.0] * len(rows)

    for port in sorted(port_groups):
        rvals = port_groups[port].get("Read",  [0.0] * len(rows))
        wvals = port_groups[port].get("Write", [0.0] * len(rows))
        rs, ws = _stats(rvals), _stats(wvals)
        print(
            f"  {port:<{name_w}}"
            f"  {rs.get('mean', 0):>10.2f}  {rs.get('max', 0):>9.2f}"
            f"  {ws.get('mean', 0):>11.2f}  {ws.get('max', 0):>10.2f}"
        )
        for i, (r, w) in enumerate(zip(rvals, wvals)):
            total_read_ts[i]  += r
            total_write_ts[i] += w

    tr, tw = _stats(total_read_ts), _stats(total_write_ts)
    print()
    print(
        f"  Total Read  (all ports): mean={tr.get('mean', 0):>8.2f}  "
        f"max={tr.get('max', 0):>8.2f}  MB/s"
    )
    print(
        f"  Total Write (all ports): mean={tw.get('mean', 0):>8.2f}  "
        f"max={tw.get('max', 0):>8.2f}  MB/s"
    )
    n_samples = len(rows)
    print(f"  Samples: {n_samples}  (time-series snapshots)")


def print_memory_profile(summary_path: str) -> None:
    """Entry point: run both DPU memory and DDR bandwidth reports."""
    base_dir = Path(summary_path).parent

    print("\n[DPU Memory Stats — profile_summary.csv]")
    print_dpu_memory_stats(base_dir / "profile_summary.csv")

    print("\n[DDR Bandwidth — vitis_ai_profile.csv]")
    print_ddr_bandwidth_stats(base_dir / "vitis_ai_profile.csv")


def find_vp_trace_file(run_summary_path: str) -> str:
    """Parse xrt.run_summary to locate the VP_TRACE CSV file."""
    summary_dir = Path(run_summary_path).parent
    with open(run_summary_path) as f:
        data = json.load(f)
    for entry in data.get("files", []):
        if entry.get("type") == "VP_TRACE":
            trace_path = summary_dir / entry["name"]
            if trace_path.exists():
                return str(trace_path)
            raise FileNotFoundError(f"VP_TRACE file not found: {trace_path}")
    raise ValueError("No VP_TRACE entry in xrt.run_summary")


def parse_vart_trace(trace_path: str):
    """
    Parse MAPPING and EVENTS sections from a VTF-format vart_trace.csv.

    Returns:
        mapping: dict[int, str]  - kernel_id → subgraph_name
        events:  list[dict]      - parsed event rows
    """
    mapping = {}
    events = []

    section = None
    with open(trace_path, newline="") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            # Section headers
            if line in ("HEADER", "STRUCTURE", "MAPPING", "EVENTS"):
                section = line
                continue

            # Blank line resets section
            if not line.strip():
                continue

            if section == "MAPPING":
                parts = line.split(",", 1)
                kid = int(parts[0])
                name = parts[1].strip() if len(parts) > 1 else ""
                mapping[kid] = name if name else f"kernel_{kid}"

            elif section == "EVENTS":
                parts = line.split(",")
                if len(parts) < 10:
                    continue
                try:
                    events.append({
                        "event_id":   int(parts[0]),
                        "start_id":   int(parts[1]),
                        "timestamp":  float(parts[2]),
                        "bucket_id":  int(parts[3]),
                        "type":       parts[4].strip(),
                        "kernel_id":  int(parts[5]),
                        "pid":        int(parts[6]),
                        "batch_size": int(parts[7]),
                        "workload":   float(parts[8]),
                        "efficiency": float(parts[9]),
                        "op_num":     int(parts[10]) if len(parts) > 10 else 0,
                    })
                except (ValueError, IndexError):
                    continue

    return mapping, events


def compute_per_layer_stats(mapping, events):
    """
    Match start/end event pairs and compute per-layer statistics.

    Returns a list of dicts sorted by execution timeline order
    (first appearance of each kernel in the trace).
    """
    # Index start events by event_id
    start_events = {e["event_id"]: e for e in events if e["start_id"] == 0}

    # Per-layer accumulators
    latencies    = defaultdict(list)   # ms
    efficiencies = defaultdict(list)   # GOP/s
    workloads    = defaultdict(list)   # GOPS
    first_seen   = {}                  # kernel_id → earliest start timestamp

    for evt in events:
        if evt["start_id"] == 0:
            # Record first appearance timestamp for timeline ordering
            kid = evt["kernel_id"]
            if kid not in first_seen:
                first_seen[kid] = evt["timestamp"]
            continue

        start = start_events.get(evt["start_id"])
        if start is None:
            continue  # unmatched end event

        kid = evt["kernel_id"]
        latency_ms = evt["timestamp"] - start["timestamp"]
        latencies[kid].append(latency_ms)
        efficiencies[kid].append(evt["efficiency"])
        workloads[kid].append(evt["workload"])

    results = []
    for kid, lats in latencies.items():
        effs = efficiencies[kid]
        wls  = workloads[kid]
        n    = len(lats)
        results.append({
            "kernel_id":       kid,
            "subgraph":        mapping.get(kid, f"kernel_{kid}"),
            "n_samples":       n,
            "first_seen_ts":   first_seen.get(kid, 0.0),
            "latency_mean_ms": statistics.mean(lats),
            "latency_std_ms":  statistics.stdev(lats) if n > 1 else 0.0,
            "latency_min_ms":  min(lats),
            "latency_max_ms":  max(lats),
            "efficiency_mean": statistics.mean(effs),
            "workload_gops":   statistics.mean(wls),
        })

    # Sort by the timestamp of first execution — reflects model layer order
    results.sort(key=lambda r: r["first_seen_ts"])
    return results


def print_results(results, mapping):
    # Column widths
    name_w = max(len(r["subgraph"]) for r in results) + 2

    header = (
        f"{'#':>3}  {'Subgraph':<{name_w}} {'N':>5}  "
        f"{'Lat mean(ms)':>12}  {'Lat std(ms)':>11}  {'CoV(%)':>7}  "
        f"{'Lat min(ms)':>11}  {'Lat max(ms)':>11}  "
        f"{'Eff mean(GOP/s)':>15}  {'Workload(GOPS)':>14}"
    )
    print(header)
    print("-" * len(header))

    total_mean_latency = sum(r["latency_mean_ms"] for r in results)

    for idx, r in enumerate(results, start=1):
        pct = 100.0 * r["latency_mean_ms"] / total_mean_latency if total_mean_latency else 0
        cov = 100.0 * r["latency_std_ms"] / r["latency_mean_ms"] if r["latency_mean_ms"] else 0
        print(
            f"{idx:>3}  {r['subgraph']:<{name_w}} {r['n_samples']:>5}  "
            f"{r['latency_mean_ms']:>12.4f}  {r['latency_std_ms']:>11.4f}  {cov:>7.1f}  "
            f"{r['latency_min_ms']:>11.4f}  {r['latency_max_ms']:>11.4f}  "
            f"{r['efficiency_mean']:>15.2f}  {r['workload_gops']:>14.4f}  ({pct:.1f}%)"
        )

    print()
    print(f"Total mean latency (sum of layers): {total_mean_latency:.3f} ms")
    print(f"Total layers: {len(results)}")
    if results:
        total_samples = results[0]["n_samples"]
        print(f"Inference runs: {total_samples}")


def save_csv(results, output_path: str):
    fieldnames = [
        "kernel_id", "subgraph", "n_samples",
        "latency_mean_ms", "latency_std_ms", "latency_cov_pct", "latency_min_ms", "latency_max_ms",
        "efficiency_mean_gops", "workload_gops",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            cov = 100.0 * r["latency_std_ms"] / r["latency_mean_ms"] if r["latency_mean_ms"] else 0
            writer.writerow({
                "kernel_id":            r["kernel_id"],
                "subgraph":             r["subgraph"],
                "n_samples":            r["n_samples"],
                "latency_mean_ms":      round(r["latency_mean_ms"], 6),
                "latency_std_ms":       round(r["latency_std_ms"], 6),
                "latency_cov_pct":      round(cov, 2),
                "latency_min_ms":       round(r["latency_min_ms"], 6),
                "latency_max_ms":       round(r["latency_max_ms"], 6),
                "efficiency_mean_gops": round(r["efficiency_mean"], 4),
                "workload_gops":        round(r["workload_gops"], 6),
            })
    print(f"\nResults saved to: {output_path}")


# ---------------------------------------------------------------------------
# Layer latency figure
# ---------------------------------------------------------------------------

def plot_layer_latency(results: list, out_dir: Path) -> None:
    """
    Horizontal bar chart of per-subgraph mean latency.

    Each bar shows mean ± std-dev, colour-coded by DPU efficiency (GOP/s)
    using a red→yellow→green colormap.  The percentage of total latency is
    annotated at the right end of each bar.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Warning: matplotlib not available — skipping layer latency plot.")
        return

    n         = len(results)
    labels    = [r["subgraph"]        for r in results]
    means     = np.array([r["latency_mean_ms"] for r in results])
    stds      = np.array([r["latency_std_ms"]  for r in results])
    effs      = np.array([r["efficiency_mean"] for r in results])
    total_lat = means.sum()
    n_runs    = results[0]["n_samples"]

    norm   = mcolors.Normalize(vmin=effs.min(), vmax=effs.max())
    cmap   = cm.RdYlGn
    colors = [cmap(norm(e)) for e in effs]

    fig_h = max(5.0, 0.38 * n + 1.8)
    fig, ax = plt.subplots(figsize=(11, fig_h))

    y = np.arange(n)
    ax.barh(
        y, means, xerr=stds, color=colors, height=0.72, align="center",
        error_kw={"elinewidth": 0.9, "capsize": 2.5, "ecolor": "#555555"},
    )

    # Percentage annotation at right end of each bar
    for i, (m, s) in enumerate(zip(means, stds)):
        pct = 100.0 * m / total_lat
        ax.text(
            m + s + total_lat * 0.005, i,
            f"{pct:.1f}%", va="center", ha="left", fontsize=6.5, color="#333333",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()   # first layer at top
    ax.set_xlabel("Mean latency (ms)", fontsize=9)
    ax.set_title(
        "Per-layer DPU latency  ·  colour = efficiency (GOP/s)",
        fontsize=10, pad=9,
    )
    ax.tick_params(axis="x", labelsize=7.5)
    ax.set_xlim(0, (means + stds).max() * 1.22)

    # Dashed mean-of-means reference line
    layer_mean = total_lat / n
    ax.axvline(layer_mean, color="#555555", lw=0.9, ls="--", alpha=0.75)
    ax.text(
        layer_mean + total_lat * 0.003, n - 0.6,
        f"avg={layer_mean:.3f} ms", fontsize=6.5, color="#555555", va="bottom",
    )

    # Colour bar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.016)
    cbar.set_label("DPU efficiency (GOP/s)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # Summary box
    summary = (
        f"Layers: {n}   Runs: {n_runs}\n"
        f"Total latency: {total_lat:.3f} ms\n"
        f"Efficiency: {effs.min():.0f} – {effs.max():.0f} GOP/s"
    )
    ax.text(
        0.985, 0.015, summary, transform=ax.transAxes,
        fontsize=6.5, va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc", alpha=0.88),
    )

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = out_dir / f"layer_latency.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=150)
        print(f"Saved {out}")

    plt.close(fig)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Parse vart_trace.csv and compute per-layer latency/efficiency stats."
    )
    parser.add_argument(
        "run_summary",
        nargs="?",
        default="xrt.run_summary",
        help="Path to xrt.run_summary (default: xrt.run_summary)",
    )
    parser.add_argument(
        "--trace",
        help="Path to vart_trace.csv directly (overrides run_summary lookup)",
    )
    parser.add_argument(
        "--csv-out",
        default=None,
        help="Save results to this CSV file",
    )
    parser.add_argument(
        "--memory-profile",
        action="store_true",
        help="Also display DPU Mem IO/BW (profile_summary.csv) and DDR bandwidth (vitis_ai_profile.csv)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate layer latency bar chart (PDF + PNG)",
    )
    parser.add_argument(
        "--plot-out",
        type=Path,
        default=None,
        help="Directory for figure output (default: figures/ next to run_summary)",
    )
    args = parser.parse_args()

    # Resolve trace file path
    if args.trace:
        trace_path = args.trace
    else:
        summary_path = args.run_summary
        if not os.path.exists(summary_path):
            print(f"Error: {summary_path} not found", file=sys.stderr)
            sys.exit(1)
        trace_path = find_vp_trace_file(summary_path)

    print(f"Parsing: {trace_path}")
    mapping, events = parse_vart_trace(trace_path)

    print(f"Found {len(mapping)} mapped kernels, {len(events)} events\n")

    results = compute_per_layer_stats(mapping, events)

    if not results:
        print("No event pairs found.")
        sys.exit(1)

    print_results(results, mapping)

    if args.csv_out:
        save_csv(results, args.csv_out)

    if args.memory_profile:
        summary_path = args.run_summary if not args.trace else None
        if summary_path and os.path.exists(summary_path):
            print("\n--- Memory Profile Report ---")
            print_memory_profile(summary_path)
        else:
            print("\nWarning: --memory-profile requires xrt.run_summary; skipping memory report.")

    if args.plot:
        print("\n--- Layer Latency Plot ---")
        plot_out = args.plot_out or Path(args.run_summary).parent / "figures"
        plot_layer_latency(results, plot_out)


if __name__ == "__main__":
    main()
