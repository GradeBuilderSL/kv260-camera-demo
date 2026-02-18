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


if __name__ == "__main__":
    main()
