# Zynq UltraScale+ MPSoC — DDRC Port Assignments

Explains which DDR controller ports correspond to which system masters, and
documents the correction made to `plot_ddr_traffic.py` and `sync_timestamps.py`.

## Background

The Zynq UltraScale+ MPSoC DDR controller (DDRC) exposes **six subordinate
AXI ports** (S0–S5).  Vitis AI's `vitis_ai_profile.csv` reports per-port
read/write bandwidth as `DDRC_PORT_S{n}_Read` / `DDRC_PORT_S{n}_Write`
columns.  Correctly labelling which port belongs to the DPU vs. the CPU is
essential for interpreting the memory traffic figures.

## Port Assignment Table

| Port | Connected master | Path | Domain |
|------|-----------------|------|--------|
| **S0** | RPU (Cortex-R5F) | Direct AXI path | LPD |
| **S1** | APU (Cortex-A53) | Via CCI-400 coherent interconnect | FPD |
| **S2** | LPD masters (GDMA, LPD-DMA, peripherals) | Via CCI-400 | LPD |
| **S3** | HP0 / HP1 FPD high-performance PL ports | S_AXI_HP0_FPD / HP1_FPD | PL → FPD |
| **S4** | HP2 / HP3 FPD high-performance PL ports | S_AXI_HP2_FPD / HP3_FPD | PL → FPD |
| **S5** | HPC0 / HPC1 (coherent PL ports via CCI) | S_AXI_HPC0_FPD / HPC1_FPD | PL → FPD |

### DPU (DPUCZDX8G) connection

The DPUCZDX8G IP core connects its AXI master ports to the PS HP slave ports:

```
DPU M_AXI_HP0  →  PS S_AXI_HP0_FPD  →  DDRC S3
DPU M_AXI_HP2  →  PS S_AXI_HP2_FPD  →  DDRC S4
```

S3 and S4 therefore carry **DPU traffic** (feature-map loads/stores and
weight loads).

### APU (Linux / Python benchmark) connection

The Cortex-A53 application processors access DDR through the CCI-400
cache-coherent interconnect:

```
APU Cortex-A53  →  CCI-400  →  DDRC S1
LPD DMA / RPU   →  CCI-400  →  DDRC S2
```

S1 and S2 carry **CPU and system DMA traffic**.

## Correction in the Codebase

The original code in `plot_ddr_traffic.py` and `sync_timestamps.py` had the
grouping **backwards**:

```python
# WRONG (original)
raw['dpu_read'] = raw['DDRC_PORT_S1_Read'] + raw['DDRC_PORT_S2_Read']
raw['cpu_read'] = raw['DDRC_PORT_S3_Read'] + raw['DDRC_PORT_S4_Read']
```

Corrected to:

```python
# CORRECT
raw['dpu_read'] = raw['DDRC_PORT_S3_Read'] + raw['DDRC_PORT_S4_Read']
raw['cpu_read'] = raw['DDRC_PORT_S1_Read'] + raw['DDRC_PORT_S2_Read']
```

This also matches empirical evidence from the benchmark data: during DPU
inference, S3 and S4 show burst reads of **350–460 MB/s** (consistent with
DPU streaming weights + feature maps), while S1 shows steady ~100 MB/s
(consistent with an APU running a Python loop).

## Empirical Cross-Check

From a representative `vitis_ai_profile.csv` sample during active inference:

| Port | Read (MB/s) | Write (MB/s) | Interpretation |
|------|-------------|--------------|----------------|
| S1   | ~120        | ~2           | APU (Python loop, cache hits) |
| S2   | ~110        | ~30          | LPD DMA / system traffic |
| S3   | ~400        | ~220         | **DPU — weight + feature-map load** |
| S4   | ~390        | ~55          | **DPU — feature-map store** |
| S5   | 0           | 0            | Unused in this design |

The high write bandwidth on S3 is consistent with the DPU storing output
feature maps back to DDR after each subgraph.

## Files Affected

| File | Change |
|------|--------|
| `utils/plot_ddr_traffic.py` | Fixed port grouping + updated legend labels |
| `utils/sync_timestamps.py`  | Fixed port grouping + added per-group stats to DDR report |

## References

- [UG1085 — Zynq UltraScale+ Device Technical Reference Manual](https://docs.amd.com/r/en-US/ug1085-zynq-ultrascale-trm/DDR-Memory-Controller)
  — DDR Memory Controller chapter: port S0–S5 master mapping
- [PG338 — DPUCZDX8G Product Guide](https://docs.amd.com/r/en-US/pg338-dpu/Connecting-the-DPUCZDX8G-to-the-Processing-System-in-the-Zynq-UltraScale-MPSoC)
  — "Connecting the DPUCZDX8G to the Processing System": HP port wiring
- [PG201 — Zynq UltraScale+ MPSoC Processing System Product Guide](https://docs.amd.com/r/en-US/pg201-zynq-ultrascale-plus-processing-system)
  — AXI slave port descriptions (HP0–HP3, HPC0–HPC1)
- [Zynq UltraScale+ MPSoC Cache Coherency — Xilinx Wiki](https://xilinx-wiki.atlassian.net/wiki/spaces/A/pages/18842098/Zynq+UltraScale+MPSoC+Cache+Coherency)
  — CCI-400 topology and which masters use coherent paths
- [Evaluating High-Performance Ports — Embedded Design Tutorials](https://xilinx.github.io/Embedded-Design-Tutorials/docs/2021.2/build/html/docs/User_Guides/SPA-UG/docs/6-evaluating-high-performance-ports.html)
  — HP vs HPC port performance comparison on MPSoC
