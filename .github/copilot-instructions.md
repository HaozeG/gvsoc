# SoftHier Copilot Instructions

This file guides AI coding agents working on the SoftHier hardware simulator. For complete environment and build details, see [CLAUDE.md](../CLAUDE.md).

## Quick Start

**Before any work:**
```bash
source sourceme.sh              # Standard GVSoC + DRAMSys environment
source sourceme_deeploy.sh      # If working on softhier_deeploy branch
```

**One-time setup:**
```bash
make softhier_preparation       # Downloads SystemC, DRAMSys, RISC-V toolchain
```

**Common build/run commands:**
```bash
make hw       # Build hardware GVSoC model
make sw       # Compile RISC-V software
make hs run   # Hardware + software + simulate
make rund     # Run with debug traces
```

Requirements: GCC ≥ 11.2.0, CMake ≥ 3.18.1, Python ≥ 3.10.3

## Hardware Simulator Architecture

SoftHier is a **GVSoC simulation model** of a softly-hierarchical many-core accelerator. Core design principle:
- **Hardware components** are modeled as **Python class hierarchies** using `gvsoc.core.models`
- Each Python descriptor instantiates a **C++ simulation engine** compiled as a shared library
- This enables flexible, configurable SystemC-based simulation of manycore clusters

### System Components

**Cluster Grid** (configurable X×Y, default 4×4 in `soft_hier/flex_cluster/flex_cluster_arch.py`):
- **Snitch RISC-V cores** (rv32imafd, 3 per cluster default) — one acts as **DM (domain master)**
- **Spatz vector unit** (optional per-core) — 8 VLSU ports, 8 FUs
- **LightRedMulE** — reconfigurable GEMM accelerator (128×32 systolic, 3-stage pipeline)
- **TCDM** — L1 scratchpad (1 MB, 128 banks, 32-bit width) with hardware interleaver
- **iDMA** — intelligent DMA (16 outstanding txns, 256-burst)
- **Transpose Engine** + **HWPE Interleaver** — specialized data movers

**Interconnect:**
- **FlexMeshNoC** — 2D mesh NoC (512-bit link width, 64 outstanding requests)
- Routes inter-cluster traffic; connects to HBM controllers at grid edges

**Memory System:**
- **L3 / HBM Controllers** — bank interleaving + scrambling
- **ZeroMem** — fast zero initialization (128 KB per node)
- **Stack** — per-core local stack (128 KB)

### Key Files & Responsibilities

| File/Directory | Purpose |
|---|---|
| `soft_hier/flex_cluster/flex_cluster.py` | Top-level FlexClusterSystem orchestration |
| `soft_hier/flex_cluster/flex_cluster_arch.py` | **Architecture parameters** — all `ARCH_*` macros derived from here |
| `soft_hier/flex_cluster/cluster_unit.py` | Per-cluster instantiation logic |
| `soft_hier/flex_cluster/flex_mesh_noc.py` | 2D mesh NoC implementation |
| `soft_hier/flex_cluster/hbm_ctrl.{py,cpp}` | HBM controller with bank interleaving |
| `soft_hier/flex_cluster/light_redmule.{py,cpp}` | GEMM accelerator (systolic array) |
| `soft_hier/flex_cluster/ctrl_registers.{py,cpp}` | Cluster control registers |
| `soft_hier/flex_cluster/util_dumpper.{py,cpp}` | Performance metric collection |

**Config Propagation:** Always run `make config` after editing `flex_cluster_arch.py` to regenerate `flex_cluster_arch.h` in the PULP chip directory.

### SDK & Runtime

**Linker & CRT:**
- `soft_hier/flex_cluster_sdk/runtime/flex_memory.ld` — Memory map (L1/L3/HBM sections)
- `soft_hier/flex_cluster_sdk/runtime/flex_start.s` — CRT0: register init, per-core stack setup

**Key Runtime Headers:**
- `flex_runtime.h` — cluster/core ID, position navigation, `flex_alloc_init()`
- `flex_alloc.h` — dual allocators: `flex_l1_malloc()` / `flex_hbm_malloc()`
- `flex_cluster_arch.h` — auto-generated `#define`s from `flex_cluster_arch.py`
- `flex_dma_pattern.h` — DMA transfer helpers

**Memory Map (per cluster):**
| Region | Base | Size |
|---|---|---|
| L1 TCDM (local) | `0x00000000` | 1 MB |
| Remote TCDM | `0x30000000` | — |
| Cluster registers | `0x20000000` | 0x200 |
| RedMulE registers | `0x20020000` | 0x200 |
| Stack | `0x10000000` | 128 KB |
| L3 / instruction mem | `0x80000000` | — |
| HBM (west → north → east → south) | `0xc0000000` | 2 MB × N nodes |

HBM is treated as a **flat pool** at runtime; address decoding routes allocations to physical nodes.

### Runtime Programming Model

- **Parallelism** via `flex_get_cluster_id()` and `flex_get_core_id()`
- **DM core** (highest hartid) orchestrates DMA and HWPE; compute cores run TCDM-local work
- **Barrier patterns** via `flex_is_first_core()` / `flex_is_dm_core()` guards
- **Navigation:** `FlexPosition` provides 2D grid helpers (`get_pos()`, `right_pos()`, `left_pos()`, etc.)
- **Allocation:** `flex_alloc_init()` must be called before using heap allocators

### Application Development

**Structure:** An application is a directory with `CMakeLists.txt` setting `SRC_SOURCES` and optionally `INCLUDE_DIRS`.

**Compilation:** RISC-V toolchain (`riscv32-unknown-elf-gcc`) with flags:
```
-O3 -ffast-math -march=rv32imafd_zfh
```
(or `rv32imafdv_zfh` with Spatz)

**Build outputs:**
- `sw_build/softhier.elf` — executable
- `sw_build/softhier.dump` — disassembly

**Constraints:** No `ebreak` in binary (enforced by build check).

### Custom Build Variables

Pass architecture/app/preload at build time:
```bash
cfg=examples/SoftHier/config/arch_test.py \
  app=examples/SoftHier/software/test \
  pld=<binary.bin> \
  make hs run
```

## Development Patterns

### Modifying Architecture

1. Edit `soft_hier/flex_cluster/flex_cluster_arch.py` (architecture parameters)
2. Run `make config` to regenerate `flex_cluster_arch.h`
3. Rebuild affected components

### Adding/Modifying Hardware Components

1. Create `.py` descriptor in `soft_hier/flex_cluster/` (GVSoC model class)
2. Pair with `.cpp` simulation engine (SystemC-based)
3. Instantiate in parent component (e.g., `flex_cluster.py`)
4. Rebuild with `make hw` or `make build`

### Testing

Tests use the **plptest framework** (declared in `testset.cfg`):
```bash
plptest --test-name <name>     # Run individual test
```

Test aggregation: `docs/developer/tutorials/testset.cfg` + `examples/testset.cfg`

CI (`.github/workflows/build.yml`) exercises:
- Basic arch tests
- Spatz vector unit checks
- Trace generation at different NoC widths
- Scale tests (4×4 to 32×32)
- HBM preload/postload
- Tutorial applications

## Code Search Strategy

When exploring the codebase, prioritize:
1. **Architecture definitions** — `flex_cluster_arch.py` (single source of truth)
2. **Component structure** — `.py` files in `soft_hier/flex_cluster/` (Python models)
3. **Simulation engines** — `.cpp` files in `soft_hier/flex_cluster/` (SystemC implementations)
4. **SDK headers** — `soft_hier/flex_cluster_sdk/runtime/include/` (API contracts)
5. **Examples** — `examples/SoftHier/` (reference implementations)

## Common Pitfalls

- **Forgetting `make config`** after architecture changes — regenerated header won't reflect changes
- **Stack overflow** — carefully manage stack size per core in linker script
- **TCDM conflicts** — verify bank interleaving doesn't cause collisions
- **HBM node addressing** — ensure allocations respect node boundaries (handled by `hbm_ctrl`)
- **No `ebreak`** — debug breakpoints will fail the build check

## References

- **Full setup/build guide:** [CLAUDE.md](../CLAUDE.md)
- **Detailed architecture:** See CLAUDE.md § Architecture Overview
- **Runtime headers:** `soft_hier/flex_cluster_sdk/runtime/include/`
- **Tests:** `examples/testset.cfg`, `docs/developer/tutorials/testset.cfg`
- **Workflows:** `.github/workflows/build.yml`
