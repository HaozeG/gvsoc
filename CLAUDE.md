# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

Before anything else, source the environment (bash shell required):
```bash
source sourceme.sh           # standard GVSoC + DRAMSys environment
source sourceme_deeploy.sh   # Deeploy-specific variant (softhier_deeploy branch)
```

Requirements (checked by `check_and_set_env.sh`): GCC ≥ 11.2.0, CMake ≥ 3.18.1, Python ≥ 3.10.3.

One-time preparation (downloads SystemC, DRAMSys, RISC-V toolchain):
```bash
make softhier_preparation
```

## Build Commands

| Command | Description |
|---|---|
| `make hw` | Build hardware GVSoC model (`make config && make TARGETS=pulp.chips.flex_cluster.flex_cluster all`) |
| `make sw` | Compile RISC-V software (outputs `sw_build/softhier.elf`, `sw_build/softhier.dump`) |
| `make hs` | Hardware + software |
| `make run` | Run simulation |
| `make runv` | Run with VCD traces |
| `make rund` | Run with debug traces |
| `make build` | Build GVSoC simulator itself (CMake, installs to `./install`) |
| `make clean` | Remove `build/` and `install/` |

Use custom config/app/preload:
```bash
cfg=examples/SoftHier/config/arch_test.py app=examples/SoftHier/software/test make hs run
cfg=<arch.py> app=<path/to/app> pld=<binary.bin> make hs run
```

Parallelism controlled by `CMAKE_FLAGS ?= -j 6`.

## Architecture Overview

SoftHier is a **GVSoC simulation model** of a softly-hierarchical many-core accelerator. GVSoC models hardware components as Python class hierarchies (using `gvsoc.core.models`) that instantiate C++ simulation engines compiled as shared libraries.

### Cluster Grid

A configurable X×Y grid of clusters (default 4×4, defined in `soft_hier/flex_cluster/flex_cluster_arch.py`). Each cluster contains:
- **Snitch cores** (RISC-V rv32imafd) — default 3 per cluster, one acts as DM (domain master) core
- **Optional Spatz vector unit** — per-core, 8 VLSU ports, 8 FUs
- **LightRedMulE** — reconfigurable GEMM accelerator (128×32 systolic, 3-stage pipeline)
- **TCDM** — L1 scratchpad (1 MB, 128 banks, 32-bit width) with hardware interleaver
- **iDMA** — intelligent DMA (16 outstanding txns, 256-burst)
- **Transpose Engine** and **HWPE Interleaver**

### Interconnect

**FlexMeshNoC** — 2D mesh NoC (`soft_hier/flex_cluster/flex_mesh_noc.py`), 512-bit link width, 64 outstanding requests. Routes inter-cluster traffic. Connects to HBM controllers at grid edges.

### Memory Map (per cluster, runtime perspective)

| Region | Base | Size |
|---|---|---|
| L1 TCDM (local) | `0x00000000` | 1 MB |
| Remote TCDM | `0x30000000` | — |
| Cluster registers | `0x20000000` | 0x200 |
| RedMulE registers | `0x20020000` | 0x200 |
| ZeroMem | `0x18000000` | 128 KB |
| Stack | `0x10000000` | 128 KB |
| L3 / instruction mem | `0x80000000` | — |
| HBM | `0xc0000000` | 2 MB × N nodes |

HBM node layout: west nodes start at `0xc0000000`, each node is 2 MB (`ARCH_HBM_NODE_ADDR_SPACE`). Address order: west → north → east → south sides of the grid.

### Hardware Model Files

```
soft_hier/flex_cluster/
├── flex_cluster.py        # Top-level FlexClusterSystem
├── flex_cluster_arch.py   # Architecture parameters (all #defines generated from here)
├── cluster_unit.py        # Per-cluster instantiation
├── flex_mesh_noc.py       # 2D mesh NoC
├── hbm_ctrl.py/.cpp       # HBM controller with bank interleaving + scrambling
├── light_redmule.py/.cpp  # RedMulE GEMM accelerator
├── hwpe_interleaver.py/.cpp
├── transpose_engine.py/.cpp
├── ctrl_registers.py/.cpp
└── util_dumpper.py/.cpp   # Performance metric collection
```

Each component is a `.py` GVSoC descriptor paired with a `.cpp` simulation engine.

### SDK / Runtime

```
soft_hier/flex_cluster_sdk/runtime/
├── flex_memory.ld          # Linker script (L1/L3/HBM sections)
├── flex_memory_deeploy.ld  # Deeploy variant
├── flex_start.s            # CRT0: register init, stack setup per core
├── include/                # Standard SDK headers
└── deeploy_include/        # Deeploy-specific headers
```

Key runtime headers:
- `flex_runtime.h` — cluster/core ID, position navigation, `flex_alloc_init()`
- `flex_alloc.h` — dual allocators: `flex_l1_malloc()` / `flex_hbm_malloc()`
- `flex_cluster_arch.h` — auto-generated `#define`s from `flex_cluster_arch.py`
- `flex_dma_pattern.h` — DMA transfer helpers

### Application Build

An application is a directory with a `CMakeLists.txt` that sets `SRC_SOURCES` and optionally `INCLUDE_DIRS`. The top-level SDK CMakeLists compiles it with the RISC-V toolchain (`riscv32-unknown-elf-gcc`), links against `flex_start.s` and `flex_memory.ld`, and produces `softhier.elf`.

Compiler flags: `-O3 -ffast-math -march=rv32imafd_zfh` (or `rv32imafdv_zfh` with Spatz). A build check enforces `no ebreak` in the output binary.

## Runtime Programming Model

- Parallelism is expressed via cluster ID (`flex_get_cluster_id()`) and core ID (`flex_get_core_id()`).
- The **DM core** (last hartid in the cluster) typically orchestrates DMA and HWPE; compute cores run TCDM-local work.
- Use `flex_is_first_core()` / `flex_is_dm_core()` guards for single-core init paths.
- `FlexPosition` provides 2D grid navigation: `get_pos(cid)`, `right_pos()`, `left_pos()`, `top_pos()`, `bottom_pos()`.
- `flex_alloc_init()` must be called (by first core) before using `flex_l1_malloc` / `flex_hbm_malloc`. Both are first-fit linked-list allocators; HBM treats all N nodes as one flat pool — allocations are routed to a physical node by address decoding in `hbm_ctrl`.

## Testing

Tests use the `plptest` framework:
```bash
plptest --test-name <name>   # run individual test
```

Test suites are declared in `testset.cfg` files and aggregate `docs/developer/tutorials/testset.cfg` and `examples/testset.cfg`.

CI (`.github/workflows/build.yml`) runs on Ubuntu 22.04 and exercises: basic arch test, Spatz check, trace generation at different NoC widths, scale tests (4×4 to 32×32), HBM preload/postload, and tutorials.

## Configuration System

Architecture is configured by editing `soft_hier/flex_cluster/flex_cluster_arch.py`, then running `make config` to propagate the generated `flex_cluster_arch.h` header into the pulp chip directory. All `ARCH_*` C macros in the SDK are derived from this single Python source.

The GVSoC simulation target is `pulp.chips.flex_cluster.flex_cluster`.
