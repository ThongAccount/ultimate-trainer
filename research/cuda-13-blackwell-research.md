# CUDA 13.x (Blackwell) — Comprehensive Research Summary

## 1. Overview & Release History

CUDA 13.x is NVIDIA's toolkit series for the **Blackwell GPU microarchitecture**, successor to Hopper (datacenter) and Ada Lovelace (consumer). Named after mathematician David Blackwell, the architecture was announced at GTC 2024 (March 18, 2024) and began shipping Q4 2024.

| Release | Driver Version (Linux) | Key Focus |
|---------|----------------------|-----------|
| CUDA 13.0 GA | ≥580.65.06 | Initial Blackwell support, CCCL 3.0 |
| CUDA 13.0 Update 1 | ≥580.82.07 | Bug fixes |
| CUDA 13.0 Update 2 | ≥580.95.05 | Bug fixes |
| CUDA 13.1 GA | ≥590.44.01 | Windows driver unbundling, incremental improvements |
| CUDA 13.1 Update 1 | ≥590.48.01 | Bug fixes |
| CUDA 13.2 GA | ≥595.45.04 | Tile programming expansion to Ampere/Ada |
| CUDA 13.2 Update 1 | ≥595.58.03 | Bug fixes |
| CUDA 13.3 GA | ≥610.43.02 | CUDA Tile C++, CUDA Python 1.0, C++23 in NVCC, CCCL 3.3, CompileIQ |
| CUDA 13.3 Update 1 | ≥610.43.02 | Latest as of June 2026 |

**Important**: Starting with CUDA 13.1, the Windows display driver is **no longer bundled** with the toolkit. Users must download separately.

---

## 2. Blackwell GPU Architecture

### 2.1 Manufacturing & Packaging
- **Process**: TSMC 4NP (datacenter), TSMC 4N (consumer)
- **Packaging**: CoWoS-L 2.5D interposer for dual-die datacenter designs
- **Dual-die datacenter design**: Two GB100 dies connected via NV-HBI (NVLink-based High Bandwidth Interface) with full cache coherency
- **Total transistors**: 208 billion (dual-die GB100 package)

### 2.2 GPU Dies & Products

#### Datacenter
| Die | CUDA Cores | Tensor Cores | L2 Cache | Memory | Interface |
|-----|-----------|-------------|----------|--------|-----------|
| GB100 | 18,432 | 576 | 60 MB | HBM3E | 8192-bit |
| GB102 | Unknown | Unknown | Unknown | HBM3E | Unknown |
| GB200 (dual-die) | — | — | — | HBM3E | NVLink 5 |

**Products**: B100 (192GB HBM3E), B200 SXM (192GB HBM3E), GB200 Superchip (288GB HBM3E), HGX B200 (8-GPU), NVL72 (72-GPU rack-scale)

#### Consumer
| Die | CUDA Cores | SMs | L2 Cache | Memory | Die Size |
|-----|-----------|-----|----------|--------|----------|
| GB202 | 24,576 | 192 | 128 MB | GDDR7 512-bit | 750 mm² |
| GB203 | 10,752 | 84 | 64 MB | GDDR7 256-bit | 378 mm² |
| GB205 | 6,400 | 50 | 48 MB | GDDR7 192-bit | 263 mm² |
| GB206 | 4,608 | 36 | 32 MB | GDDR7 128-bit | 181 mm² |
| GB207 | 2,560 | 20 | 32 MB | GDDR7 128-bit | 149 mm² |
| GB10 | 6,144 | 48 | 50 MB | GDDR7 256-bit | — |

**Consumer products**: RTX 5090 (GB202), RTX 5080 (GB203), RTX 5070 Ti (GB203), RTX 5070 (GB205), RTX 5060 Ti (GB206), RTX 5050 (GB207), DGX Spark (GB10B)

---

## 3. Compute Capabilities & SM Architecture

### 3.1 Supported Architectures in CUDA 13.x (nvcc)

```
sm_75  (Turing - CC 7.5)
sm_80  (Ampere - CC 8.0)
sm_86  (Ampere - CC 8.6)
sm_87  (Ampere - CC 8.7)
sm_88  (Ada Lovelace - CC 8.8)
sm_89  (Ada Lovelace - CC 8.9)
sm_90  (Hopper - CC 9.0)
sm_90a (Hopper - architecture-specific)
sm_100  (Blackwell - CC 10.0)
sm_100f (Blackwell - with extended features)
sm_100a (Blackwell - architecture-specific)
sm_103  (Blackwell Ultra - CC 10.3)
sm_103f (Blackwell Ultra - with extended features)
sm_103a (Blackwell Ultra - architecture-specific)
sm_110  (Future - CC 11.0, possibly Rubin)
sm_110f / sm_110a
sm_120  (Future - CC 12.0)
sm_120f / sm_120a
sm_121  (Future - CC 12.1)
sm_121f / sm_121a
```

### 3.2 Architecture Variants Explained
- **`sm_XX`**: Standard (non-architecture-specific) — PTX forward-compatible within the generation
- **`sm_XXa`**: Architecture-specific — can use all hardware features of that specific GPU, but **not** forward-compatible via PTX JIT
- **`sm_XXf`**: Extended features variant (new in Blackwell era) — includes additional feature flags

### 3.3 Compute Capability Mapping
| Compute Capability | Architecture | SM Targets |
|-------------------|--------------|------------|
| CC 10.0 | Blackwell (datacenter) | sm_100, sm_100f, sm_100a |
| CC 10.3 | Blackwell Ultra | sm_103, sm_103f, sm_103a |
| CC 11.0 | Next-gen (likely Rubin) | sm_110, sm_110f, sm_110a |
| CC 12.0 | Future | sm_120, sm_120f, sm_120a |
| CC 12.1 | Future | sm_121, sm_121f, sm_121a |

**Note**: Blackwell consumer GPUs (GeForce RTX 50 series) report **CC 12.0**, while datacenter GPUs (B100/B200/GB200) report **CC 10.0**. Both are 64-bit only; 32-bit support has been removed.

---

## 4. Blackwell-Specific Hardware Features

### 4.1 Fifth-Generation Tensor Cores
- New FP4 and FP6 (microscaling) data type support via **OCP MXFP4/MXFP6** formats
- Second-generation **Transformer Engine** with dynamic quantization to MXFP4/MXFP6
- 20 petaflops of FP4 compute for the dual-GPU GB200 superchip (excluding sparsity gains)
- Improved TF32 performance on Blackwell and Blackwell Ultra

### 4.2 FP4 & Microscaling Formats
- **MXFP4**: 4-bit floating-point with shared scale factors per microblock (OCP standard)
- **MXFP6**: 6-bit floating-point with shared scale factors
- Enables sub-8-bit inference with improved accuracy vs. naive INT4 quantization
- Hardware-native support (not software emulation)

### 4.3 NVLink 5 & NV-HBI
- **NVLink 5**: Latest generation interconnect for multi-GPU scaling
- **NV-HBI (NVLink High Bandwidth Interconnect)**: Based on NVLink 7 protocol, connects two GB100 dies on the same package
- Full cache coherency between the two dies
- Estimated ~$10B R&D investment (per Jensen Huang; disputed)
- **NVL72**: 72-GPU rack-scale system using NVLink for full GPU-to-GPU connectivity

### 4.4 AI Management Processor (AMP)
- Dedicated **RISC-V** scheduler chip on the GPU
- Offloads scheduling from CPU, gives GPU more autonomous resource control
- Used through Windows Hardware-Accelerated GPU Scheduling (HAGS)

### 4.5 Fourth-Generation Ray Tracing Cores
- **Triangle Cluster Intersection Engine** for Mega Geometry
- **Linear Swept Spheres** for fine-detail ray tracing (hair, fur)

### 4.6 Memory
- **Datacenter**: HBM3E (up to 192GB per B200, 288GB per GB200 Superchip)
- **Consumer**: GDDR7 (up to 512-bit on GB202)
- PCIe 5.0 (consumer), PCIe 6.0 (datacenter)

---

## 5. NVCC Compiler Flags for CUDA 13.x

### 5.1 Architecture Flags
```bash
# Target Blackwell datacenter GPU (CC 10.0)
nvcc -arch=sm_100 file.cu              # Shorthand: compiles to sm_100 + compute_100
nvcc -arch=compute_100 -code=sm_100    # Explicit virtual + real

# Architecture-specific (all HW features, no forward compat)
nvcc -arch=sm_100a file.cu             # Uses all Blackwell-specific features

# Extended features
nvcc -arch=sm_100f file.cu

# Blackwell Ultra (CC 10.3)
nvcc -arch=sm_103 file.cu
nvcc -arch=sm_103a file.cu

# Multi-architecture fat binary
nvcc -gencode arch=compute_90,code=sm_90 \
     -gencode arch=compute_100,code=sm_100 \
     -gencode arch=compute_100,code=compute_100 file.cu

# Native detection
nvcc -arch=native file.cu              # Detect visible GPUs at compile time
```

### 5.2 Key New/Notable Flags in CUDA 13.x
| Flag | Description |
|------|-------------|
| `-arch=native` | Auto-detect visible GPU architecture |
| `-std=c++23` | **Official C++23 support** (CUDA 13.3+) |
| `--tilebc` | Tile Binary Compilation (new in 13.x for CUDA Tile) |
| `--compile-as-tools-patch` / `-astoolspatch` | Compile as tools patch |
| `--gpu-architecture` / `-arch` | Now supports sm_100, sm_103, sm_110, sm_120, sm_121 |
| Advanced Controls File (ACF) | For CompileIQ auto-tuning (Blackwell+ only) |

### 5.3 C++ Standard Support
| Flag | Status |
|------|---------|
| `--std=c++17` | Default for CCCL 3.0+ (minimum) |
| `--std=c++20` | Supported |
| `--std=c++23` | **Officially supported in CUDA 13.3+** |

---

## 6. C++23 Device Features in CUDA 13.3

CUDA 13.3 brings **official C++23 support in NVCC**. Key C++23 features now available on-device:

- `std::expected<T,E>` — Error handling without exceptions
- `std::print` / `std::println` — Formatted output (if supported on device)
- `std::mdspan` — Multidimensional array view (critical for tensor programming)
- Deducing `this` (P0847R8) — Explicit object parameters
- `if consteval` — Compile-time evaluation branching
- `std::flat_map` / `std::flat_set` — Sorted containers
- `auto(x)` — Decay-copy in constant expressions
- `std::generator` — Coroutine generator (host-side primarily)
- Ranges improvements (zip, chunk, slide, etc.)

**Important**: Not all C++23 features are available on-device. Standard library features requiring OS services (file I/O, threads, etc.) remain host-only.

---

## 7. CUDA Tile Programming (New in 13.x)

### 7.1 Overview
CUDA Tile is a **new programming model** introduced in CUDA 13.x that enables high-level, tile-based kernel development. It automatically manages:
- **Parallelism** — Thread/block scheduling
- **Memory movement** — Shared memory, global memory transfers
- **Asynchrony** — Overlapping computation and data movement
- **Architecture portability** — Optimal across GPU generations

### 7.2 CUDA Tile C++ (CUDA 13.3)
- Released in CUDA 13.3, extending Tile support to C++ (previously Python-only)
- Enables existing C++ codebases to use tile-based kernels
- Compiler support via `--tilebc` flag
- **TILE-IR AS** component version 13.3.36 in CUDA 13.3

### 7.3 cuTile Python (CUDA 13.3)
- New constructs: **closures** and **recursion** in cuTile Python
- Extended to **Ampere and Ada** architectures (previously Blackwell-only)

---

## 8. CompileIQ Auto-Tuning Framework (New in 13.x)

- **CompileIQ**: New compiler auto-tuning framework
- Delivers up to **15% speedup** on critical kernels (GEMM, attention)
- Uses **Advanced Controls Files (ACF)** for specifying tuning parameters
- ACF support is **Blackwell and later architectures only**
- Uses the `ctadvisor` component (version 13.3.33)
- Allows specifying compilation strategies, optimization hints, and kernel variants

---

## 9. CCCL 3.0 (CUDA Core Compute Libraries)

### 9.1 Major Changes in CCCL 3.0 (shipped with CUDA 13.0)
CCCL 3.0 is a **ground-up consolidation** of Thrust, CUB, and libcu++ under one repository. Key changes:

#### Requirements
- **C++17 minimum** (C++11/C++14 dropped)
- **CUDA Toolkit 12.0+** required (CTK 11.0+ dropped)
- GCC 7+, Clang 14+, MSVC 2019+
- ICC support dropped
- CUDA Dynamic Parallelism v1 (CDPv1) dropped

#### Header Directory Restructuring
```
# Before CUDA 13.0                    # After CUDA 13.0
${CTK_ROOT}/include/cuda/      →     ${CTK_ROOT}/include/cccl/cuda/
${CTK_ROOT}/include/cub/       →     ${CTK_ROOT}/include/cccl/cub/
${CTK_ROOT}/include/thrust/    →     ${CTK_ROOT}/include/cccl/thrust/
```
**Do NOT use `#include <cccl/...>`** — it still uses the old include paths.

#### Removed APIs (major)
| Removed | Replacement |
|---------|------------|
| `thrust::optional` | `cuda::std::optional` |
| `thrust::tuple` | `cuda::std::tuple` |
| `thrust::pair` | `cuda::std::pair` |
| `thrust::numeric_limits` | `cuda::std::numeric_limits` |
| `thrust::not1`, `thrust::not2` | C++ standard equivalents |
| `thrust::unary_function`, `thrust::binary_function` | Direct operator structs |
| `cub::Mutex` | `cuda::std::mutex` |
| `cub::GridBarrier` | Cooperative Groups |
| `cub::DeviceSpmv` | cuSPARSE |
| `cub::BFE` | `cuda::bitfield_extract` |
| 50+ legacy macros | Modern C++ equivalents |

#### New Features in CCCL 3.0
- `cuda::std::numeric_limits<__float128>` support
- `cuda::std::optional<T&>` (P2988)
- `cuda::std::numbers` — Mathematical constants
- **NVFP8/6/4** types in `<cuda/std/cmath>`
- `cuda::overflow_cast` — Safe numeric conversions
- `cuda::ilog2`, `cuda::ilog10` — Integer logarithms
- `cuda::round_up`, `cuda::round_down` — Alignment utilities
- `cub::DeviceSegmentedReduce` — Large segment count support
- `cub::DeviceCopy::Batched` / `cub::DeviceMemcpy::Batched` — Large buffer counts
- `thrust::offset_iterator` — New iterator type
- `par_nosync` respected for temporary storage allocations
- PDL (Programmatic Dependent Launch) enabled in triple-chevron launches

### 9.2 CCCL 3.3 (shipped with CUDA 13.3)
- **DLPack/mdspan** tensor interoperability
- CUDA Python `cuda-cccl` package on PyPI
- Python 3.14 support
- `cuda.cccl.cooperative`: Block-level sorting/data-movement for multi-dimensional thread blocks
- `cuda.cccl.parallel`: New device-level segmented-reduce, unique-by-key, merge-sort algorithms

---

## 10. Green Contexts (New Programming Model Feature)

Introduced in CUDA 13.x, **Green Contexts** allow splitting a GPU's SMs into disjoint partitions:
- Each partition gets its own context and streams
- Latency-sensitive kernels are **shielded from long-running throughput kernels** in the same process
- Available via both C API and `cuda.core` Python API

```python
# Python example (CUDA 13.3)
from cuda.core import Device, ContextOptions, SMResourceOptions
dev = Device()
sm = dev.resources.sm
long_grp, crit_grp = sm.split(SMResourceOptions(count=(sm.sm_count - 16, 16)))[0]
ctx_crit = dev.create_context(ContextOptions(resources=[crit_grp]))
s_crit = ctx_crit.create_stream()
```

---

## 11. Process Checkpointing (New)

- **Snapshot the full CUDA state** of a running process: device allocations, streams, context
- Restore later — enables CRIU-style workflows
- Use cases: fault-tolerant long jobs, preemption/migration on shared clusters, fast warm-start
- **Linux only**
- Available via `cuda.core.checkpoint` Python API

---

## 12. Changes from CUDA 12.x to 13.x

### 12.1 Architectural
| Feature | CUDA 12.x | CUDA 13.x |
|---------|-----------|-----------|
| Max Compute Capability | CC 9.0+ (Hopper) | CC 10.0–12.1 (Blackwell+) |
| 32-bit support | Supported | **Removed** (64-bit only) |
| Turing (sm_75) | Full support | Supported but deprecated path |
| Header layout | Direct in `include/` | CCCL headers in `include/cccl/` |

### 12.2 Compiler
| Feature | CUDA 12.x | CUDA 13.x |
|---------|-----------|-----------|
| C++ standard | C++20 max | **C++23** officially supported |
| CCCL version | 2.x | **3.0+** |
| Minimum C++ for CCCL | C++11 | **C++17** |
| Tile programming | N/A | **CUDA Tile** (C++ & Python) |
| CompileIQ | N/A | Auto-tuning framework (Blackwell+) |
| ACF support | N/A | Advanced Controls Files |
| Architecture-specific `a` suffix | sm_90a | sm_100a, sm_103a, sm_110a, etc. |
| Extended features `f` suffix | N/A | sm_100f, sm_103f, sm_110f, etc. |
| Native arch detection | N/A | `-arch=native` |

### 12.3 Libraries
| Library | Key Changes |
|---------|------------|
| cuBLAS | FP4 matmul support, Blackwell Ultra perf improvements, TF32 improvements |
| cuSPARSE | Performance improvements for Blackwell |
| cuSOLVER | 64-bit `cusolverDnXpolar`, improved `cusolverDnXgetrf` for sm_100/sm_103/sm_120 |
| cuFFT | Expanded LTO callbacks |
| NPP | 13.1.2.81 |
| nvJPEG | Multiple updates across 13.x |

### 12.4 Platform
| Feature | CUDA 12.x | CUDA 13.x |
|---------|-----------|-----------|
| Windows driver | Bundled | **Unbundled** (from 13.1) |
| Nsight Eclipse | Included | **Removed** (from 13.3) |
| CUDA Python | Experimental | **1.0 stable** |
| IPC in Python | N/A | Native GPU memory sharing |
| Green Contexts | N/A | SM partitioning |
| Process Checkpoint | N/A | Full CUDA state snapshot |

---

## 13. CUDA 13.x Component Versions (13.3 Update 1)

| Component | Version |
|-----------|---------|
| NVCC | 13.3.73 |
| CUDA Runtime | 13.3.29 |
| Thrust | 3.3.3 |
| CUB | 3.3.3 |
| libcu++ | 3.3.3 |
| Cooperative Groups | 13.3.3.4.1 |
| cuBLAS | 13.6.0.2 |
| cuFFT | 12.3.0.29 |
| cuSOLVER | 12.2.6.9 |
| cuSPARSE | 12.8.2.51 |
| cuRAND | 10.4.3.29 |
| NPP | 13.1.2.81 |
| NVRTC | 13.3.33 |
| CompileIQ (ctadvisor) | 13.3.33 |
| TILE-IR AS | 13.3.36 |
| CUPTI | 13.3.75 |
| Compute Sanitizer | 13.3.75 |
| nvFatbin | 13.3.29 |

---

## 14. Impact on ML/AI Training (for ultimate-ai-model project)

### 14.1 What CUDA 13.x + Blackwell Means for Our Project

1. **FP4 Tensor Core Operations**: Blackwell's MXFP4 support can dramatically speed up inference and potentially training through lower-precision computation with maintained accuracy via microscaling

2. **5th-gen Tensor Cores**: Faster matrix operations across all precisions (FP16, BF16, TF32, FP8, FP6, FP4)

3. **CCCL 3.0 Migration Required**:
   - Must use C++17 minimum
   - Replace `thrust::optional` → `cuda::std::optional`
   - Replace `thrust::tuple` → `cuda::std::tuple`
   - Replace `thrust::pair` → `cuda::std::pair`
   - Update include paths if using custom build systems
   - Drop any CDPv1 usage

4. **Green Contexts**: Can partition SMs for training vs. inference on the same GPU

5. **CUDA Tile**: New high-level programming model for tile-based kernels — could simplify custom kernel development

6. **CompileIQ**: Auto-tuning for custom kernels (GEMM, attention) — up to 15% speedup without manual tuning

7. **Architecture targeting**:
   - For B100/B200: `-arch=sm_100` (or `sm_100a` for full features)
   - For Blackwell Ultra: `-arch=sm_103`
   - For RTX 5090: `-arch=sm_120` (consumer CC)
   - Multi-arch: `-gencode=arch=compute_100,code=sm_100 -gencode=arch=compute_90,code=sm_90`

### 14.2 Recommended Build Configuration
```bash
nvcc -std=c++17 \
     -arch=sm_100a \
     --optimize=3 \
     -Xcompiler -fPIC \
     -ccbin g++-13 \
     --threads=0 \
     file.cu
```

---

## 15. Key References

1. NVIDIA CUDA Toolkit 13.3 Release Notes: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
2. CUDA C++ Programming Guide v13.3: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
3. NVCC Compiler Driver v13.3: https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html
4. CUDA 13.3 Blog Post: https://developer.nvidia.com/blog/nvidia-cuda-13-3-enhances-gpu-development-with-tile-programming-in-c-compiler-autotuning-and-python-updates/
5. CCCL 3.0 Release: https://github.com/NVIDIA/cccl/releases/tag/v3.0.0
6. CCCL 3.3 Release: https://github.com/NVIDIA/cccl/releases/tag/v3.3.0
7. Blackwell Architecture (Wikipedia): https://en.wikipedia.org/wiki/Blackwell_(microarchitecture)
8. CUDA Wikipedia: https://en.wikipedia.org/wiki/CUDA
