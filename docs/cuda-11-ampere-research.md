# CUDA 11.x (Ampere Architecture) — Comprehensive Research Summary

## Table of Contents
1. [Architecture Overview](#1-architecture-overview)
2. [SM Architecture & Compute Cores](#2-sm-architecture--compute-cores)
3. [Memory Hierarchy](#3-memory-hierarchy)
4. [Third-Generation Tensor Cores](#4-third-generation-tensor-cores)
5. [New Data Types: TF32, BFloat16, FP64 Tensor](#5-new-data-types-tf32-bfloat16-fp64-tensor)
6. [Fine-Grained Structured Sparsity (2:4)](#6-fine-grained-structured-sparsity-24)
7. [Multi-Instance GPU (MIG)](#7-multi-instance-gpu-mig)
8. [Cooperative Groups & Thread Collectives](#8-cooperative-groups--thread-collectives)
9. [Dynamic Parallelism](#9-dynamic-parallelism)
10. [CUDA Graphs & Task Graph Acceleration](#10-cuda-graphs--task-graph-acceleration)
11. [Unified Memory Improvements](#11-unified-memory-improvements)
12. [Stream-Ordered Memory Allocator](#12-stream-ordered-memory-allocator)
13. [Asynchronous Operations & Barriers](#13-asynchronous-operations--barriers)
14. [L2 Cache Management & Residency Controls](#14-l2-cache-management--residency-controls)
15. [nvcc Compilation Flags for CUDA 11.x](#15-nvcc-compilation-flags-for-cuda-11x)
16. [C++17 Device Features](#16-c17-device-features)
17. [CUDA 11.x Minor Version Features](#17-cuda-11x-minor-version-features)
18. [Performance Specifications](#18-performance-specifications)
19. [Implications for Our ML Trainer](#19-implications-for-our-ml-trainer)

---

## 1. Architecture Overview

### GA100 GPU — The Ampere Data Center Die

**Fabrication**: TSMC 7nm N7 process, 54.2 billion transistors, 826 mm² die

**Full GA100 GPU (128 SMs)**:
- 8 GPCs (Graphics Processing Clusters)
- 8 TPCs per GPC, 2 SMs per TPC, 16 SMs per GPC
- 64 FP32 CUDA Cores per SM → 8,192 total FP32 cores
- 4 third-gen Tensor Cores per SM → 512 total Tensor Cores
- 6 HBM2 stacks, 12 × 512-bit memory controllers

**A100 GPU Implementation (108 SMs)**:
- 7 GPCs, 7–8 TPCs per GPC
- 6,912 FP32 CUDA Cores
- 432 Tensor Cores
- 5 HBM2 stacks, 10 × 512-bit memory controllers
- **Compute Capability**: 8.0 (`sm_80`)

### Key Architectural Generational Leaps (V100 → A100)
| Feature | V100 (Volta) | A100 (Ampere) | Improvement |
|---------|-------------|---------------|-------------|
| Process | TSMC 12nm | TSMC 7nm | 1 node jump |
| Transistors | 21.1B | 54.2B | 2.6× |
| FP32 CUDA Cores | 5,120 | 6,912 | 1.35× |
| Tensor Cores/SM | 8 (2nd gen) | 4 (3rd gen) | 2× FMA/core |
| HBM2 Capacity | 32 GB | 40/80 GB | 1.25–2.5× |
| HBM2 Bandwidth | 900 GB/s | 1,555/2,039 GB/s | 1.7–2.3× |
| L2 Cache | 6 MB | 40 MB | 6.7× |
| L1+Shared/SM | 128 KB | 192 KB | 1.5× |
| NVLink BW | 300 GB/s | 600 GB/s | 2× |

---

## 2. SM Architecture & Compute Cores

### SM Structure (GA100/A100)

Each SM is partitioned into **4 processing blocks (sub-partitions)**, each containing:
- 16 FP32 CUDA Cores
- 16 FP32/INT32 CUDA Cores (dual-purpose)
- 1 third-gen Tensor Core
- 1 warp scheduler, 1 dispatch unit per scheduler
- 4 texture units (shared across SM)

**Per SM totals**:
- 64 FP32 CUDA Cores
- 64 INT32 CUDA Cores (can execute simultaneously with FP32)
- 4 Tensor Cores
- 4 warp schedulers, 4 dispatch units
- 16 texture units

### Simultaneous FP32 + INT32 Execution
A100 continues Volta/Turing's ability to execute FP32 and INT32 operations simultaneously at full throughput. Critical for workloads with mixed pointer arithmetic (INT32) and floating-point compute (FP32) in the same inner loop.

### Occupancy Limits (Compute Capability 8.0)
| Resource | Per SM |
|----------|--------|
| Max warps | 64 (2,048 threads) |
| Max thread blocks | 32 |
| Max 32-bit registers | 65,536 |
| Max registers per block | 65,536 |
| Max registers per thread | 255 |
| Max threads per block | 1,024 |
| Shared memory per SM | Configurable up to 164 KB |

---

## 3. Memory Hierarchy

### 3.1 Register File
- **65,536 registers per SM** (same as Volta)
- **255 registers per thread** maximum
- Fastest memory, ~1 cycle latency
- Register spilling goes to L1/local memory

### 3.2 Shared Memory + L1 Cache (Combined)

The combined L1 data cache and shared memory subsystem:
- **192 KB per SM** total (vs. 128 KB on V100) — **1.5× increase**
- **Configurable partitioning**: Up to 164 KB for shared memory, remainder for L1
  - Default: 100 KB shared + 92 KB L1
  - Maximum shared: 164 KB shared + 28 KB L1
- Shared memory bandwidth: ~17× faster than global memory
- L1 cache provides automatic caching of global memory accesses
- Supports **compute data compression** for compressible data patterns

```cpp
// Configure shared memory / L1 split
cudaFuncSetAttribute(kernel,
    cudaFuncAttributePreferredSharedMemoryCarveout,
    cudaSharedmemCarveoutMaxShared); // Maximize shared memory
// Or per-function:
cudaFuncSetAttribute(kernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    163840); // 160 KB dynamic shared
```

### 3.3 L2 Cache

**Massive L2 Cache — 40 MB** (vs. 6 MB on V100):
- Partitioned crossbar structure, 2.3× the read bandwidth of V100
- Shared across all SMs, sits outside GPCs
- **L2 Cache Residency Controls** (new in CUDA 11/Ampere):
  - `cudaAccessPropertyPersisting` — data preferentially stays in L2
  - `cudaAccessPropertyStreaming` — data evicted quickly
  - `cudaAccessPropertyNormal` — default behavior
- Set-aside portion of L2 for persistent data via `cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, ...)`
- Compute Data Compression: up to 4× improvement in DRAM bandwidth, up to 2× effective L2 capacity

### 3.4 Global Memory (HBM2)

**A100-40GB**: 40 GB HBM2, 1,555 GB/s bandwidth, 5 stacks × 8 dies
**A100-80GB**: 80 GB HBM2, 2,039 GB/s bandwidth, 5 stacks × 16 dies

- SECDED ECC protection across HBM2, L2, L1, and register file
- Supports **row remapping** for degraded cells
- **Compute Data Compression**: transparent hardware compression up to 4× DRAM bandwidth improvement

### 3.5 Memory Hierarchy Summary
```
Registers:      ~0 cycles, 255/thread, 65K/SM
Shared Memory:  ~20-30 cycles, up to 164 KB/SM, programmer-managed
L1 Cache:       ~30 cycles, combined with shared, 192 KB/SM total
L2 Cache:       ~200 cycles, 40 MB, shared across GPU, residency controls
HBM2:           ~400-600 cycles, 40-80 GB, 1.5-2 TB/s
```

---

## 4. Third-Generation Tensor Cores

### Architecture
- **4 Tensor Cores per SM** (vs. 8 in Volta, but each is 4× more powerful)
- Each performs **256 FP16 FMA ops/clock** (vs. 64 in Volta)
- Total per SM: 1,024 dense FP16 FMA ops/clock — **2× per SM vs. Volta**

### Supported Operations
| Data Type | Accumulator | A100 Dense TFLOPS | A100 Sparse TFLOPS |
|-----------|------------|-------------------|-------------------|
| FP64 | FP64 | 19.5 | — |
| TF32 | FP32 | 156 | 312 |
| FP16 | FP16 | 312 | 624 |
| FP16 | FP32 | 312 | 624 |
| BF16 | FP32 | 312 | 624 |
| INT8 | INT32 | 624 TOPS | 1,248 TOPS |
| INT4 | INT32 | 1,248 TOPS | 2,496 TOPS |
| Binary | INT32 | 4,992 TOPS | — |

### WMMA (Warp Matrix Multiply-Accumulate) API
```cpp
#include <mma.h>
using namespace nvcuda::wmma;

// A100 supports larger tiles: 64×64×16, 64×32×16, 32×64×16, etc.
// New in CUDA 11 for sm_80:
fragment<matrix_a, 64, 64, 16, bf16_t, row_major> a_frag;
fragment<matrix_b, 64, 64, 16, bf16_t, col_major> b_frag;
fragment<accumulator, 64, 64, 16, float> c_frag;

fill_fragment(c_frag, 0.0f);
mma_sync(c_frag, a_frag, b_frag, c_frag); // Tensor Core MMA
```

### PTX MMA Instructions (Direct Access)
```ptx
// New in sm_80: .sp modifier for sparse MMA
wmma.mma.sync.aligned.m16n16k16.row.col.f32.tf32.tf32.f32
wmma.mma.sync.aligned.m16n16k16.row.col.f32.bf16.bf16.f32
wmma.mma.sync.aligned.m16n16k16.row.col.f64.f64.f64.f64  // FP64 tensor!
```

---

## 5. New Data Types: TF32, BFloat16, FP64 Tensor

### TF32 (TensorFloat-32)
- **Format**: 1 sign + 8 exponent (FP32 range) + 10 mantissa (FP16 precision) = 19 bits
- **Usage**: Default math mode for Tensor Cores on A100
- **Key property**: Reads FP32 inputs, uses FP32 range, reduced internal precision, outputs FP32
- **No code changes needed** — automatic acceleration in frameworks (PyTorch default)
- **Performance**: 10× faster than V100 FP32 FMA; 20× with sparsity
- **When to use**: General DL training (default), HPC (good precision, great speed)

```cpp
// TF32 is implicit in Tensor Core ops by default on sm_80
// In cuBLAS, it's automatic. To explicitly control:
cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
```

### BFloat16 (BF16)
- **Format**: 1 sign + 8 exponent (same range as FP32) + 7 mantissa = 16 bits
- **Key property**: Same dynamic range as FP32, lower precision — stable training without loss scaling
- **Performance**: Same throughput as FP16 Tensor Core ops (312 TFLOPS dense)
- **CUDA type**: `__nv_bfloat16` in `<cuda_bf16.h>`

```cpp
#include <cuda_bf16.h>

__nv_bfloat16 a_bf16 = __float2bfloat16(3.14f);
float f = __bfloat162float(a_bf16);

// In PyTorch:
// model = model.bfloat16()  // Convert model to BF16
// with torch.autocast('cuda', dtype=torch.bfloat16):
//     output = model(input)
```

### FP64 Tensor Core (New!)
- IEEE-compliant double-precision matrix operations on Tensor Cores
- **19.5 TFLOPS** (2.5× V100's 7.8 TFLOPS FP64)
- Each Tensor Core computes 4 FP64 DFMA ops/clock
- Each SM: 64 FP64 FMA ops/clock (2× V100)
- Replaces 8 DFMA instructions from V100 with 1 instruction
- Critical for HPC: climate modeling, molecular dynamics, CFD

---

## 6. Fine-Grained Structured Sparsity (2:4)

### Concept
- **2:4 structured sparsity pattern**: Exactly 2 non-zero values per every 4-element vector
- 50% sparsity with hardware acceleration → **2× throughput boost**
- Compressed storage reduces memory footprint and bandwidth by ~2×

### How It Works
1. **Train** a dense network normally
2. **Prune** weights using 2:4 structured pattern (keep 2 largest of each 4)
3. **Fine-tune** remaining non-zero weights (few epochs)
4. **Deploy** with Sparse Tensor Core acceleration

### Hardware Support
- Sparse Tensor Core instructions skip zero-value computations entirely
- Compression metadata (indices of non-zero elements) used to align operands
- Supported for FP16, BF16, TF32, INT8 Tensor Core operations

### Usage
```cpp
// In cuBLAS (CUDA 11+):
cublasLtMatmulAlgo_t algo;
// Enable sparsity in matrix descriptor
cublasLtMatrixLayoutSetAttribute(
    layoutB, CUBLASLT_MATRIX_LAYOUT_IS_2_4_INFERENCE,
    &is_sparse, sizeof(is_sparse));

// PyTorch:
// torch.sparse.to_sparse_semi_structured(tensor)  # 2:4 sparsity
```

### Impact
| Operation | Dense | With 2:4 Sparsity |
|-----------|-------|--------------------|
| FP16 TC | 312 TFLOPS | 624 TFLOPS |
| TF32 TC | 156 TFLOPS | 312 TFLOPS |
| INT8 TC | 624 TOPS | 1,248 TOPS |

---

## 7. Multi-Instance GPU (MIG)

### Concept
- Physically partition a single A100 into up to **7 independent GPU instances**
- Each instance has **isolated memory paths**: crossbar ports, L2 cache banks, memory controllers, DRAM address buses — all uniquely assigned
- Guaranteed QoS, fault isolation, security isolation
- Transparent to CUDA — existing code runs unchanged

### Instance Profiles (A100-40GB)
| Profile | SMs | Memory | Instances Available |
|---------|-----|--------|-------------------|
| 1g.5gb | 14 | 5 GB | Up to 7 |
| 2g.10gb | 28 | 10 GB | Up to 3 |
| 3g.20gb | 42 | 20 GB | Up to 2 |
| 4g.20gb | 56 | 20 GB | 1 |
| 7g.40gb | 98 | 40 GB | 1 (full GPU) |

### Management
```bash
# Enable MIG mode
nvidia-smi -i 0 -mig 1

# List profiles
nvidia-smi mig -i 0 -lgip

# Create instances
nvidia-smi mig -i 0 -cgi 19,19,19,19,19,19,19 -C

# In Docker:
docker run --gpus '"device=0:0,0:1"' nvcr.io/nvidia/pytorch:...
```

### Use Cases
- Cloud multi-tenancy (CSPs get up to 7× more GPU instances)
- Multiple inference workloads at guaranteed latency
- Development + testing on shared GPU
- Mixed HPC workloads with isolation

---

## 8. Cooperative Groups & Thread Collectives

### Enhancements in CUDA 11 / Ampere

**Warp-level `reduce()` collective** (new hardware instruction on A100):
```cpp
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

auto block = cg::this_thread_block();
auto tile = cg::tiled_partition<32>(block); // warp tile

int val = data[threadIdx.x];
int sum = cg::reduce(tile, val, cg::plus<int>()); // Warp reduction
```

**`labeled_partition()` — custom non-power-of-2 partitions**:
```cpp
cg::coalesced_group active = cg::coalesced_threads();
int bucket = active.match_any(value);  // Group by value
cg::coalesced_group subgroup = cg::labeled_partition(active, bucket);

if (subgroup.thread_rank() == 0) {
    // Each unique group does its own atomic
    int pos = atomicAdd(&output[bucket], subgroup.size());
}
// Share result within subgroup via shuffle
```

**Cooperative Kernel Launch** (grid-wide synchronization):
```cpp
// Launch cooperative kernel (all SMs active simultaneously)
void* args[] = {&d_data, &d_result};
cudaLaunchCooperativeKernel((void*)kernel, gridDim, blockDim, args);

// Grid-wide sync inside kernel
cooperative_groups::grid_group grid = cooperative_groups::this_grid();
grid.sync(); // All threads in grid synchronize
```

**Thread Block Clusters** (Ampere sm_80+ concept for cluster.sync):
- New hierarchy level between block and grid
- Blocks in a cluster can access each other's shared memory
- Cluster-level synchronization

---

## 9. Dynamic Parallelism

### Overview
CUDA Dynamic Parallelism (CDP) allows GPU threads to **launch new kernels** directly from device code, without returning to the host. Available since Kepler (sm_35), refined in each generation.

### CDP on Ampere (sm_80)
- Device-side kernel launches using `kernel<<<grid, block>>>(args)`
- Nested device runtime: `<cuda_device_runtime_api.h>`
- Parent-child synchronization, events, device-side malloc
- **Performance notes on A100**:
  - Lower overhead than older architectures due to faster context switching
  - L2 cache improvements help child kernel data access
  - Generally, **CUDA Graphs are preferred** for predictable workloads (lower overhead)
  - CDP best for irregular/adaptive workloads where launch structure is data-dependent

```cpp
__global__ void parent_kernel(float* data, int n) {
    if (threadIdx.x == 0 && n > 1024) {
        // Launch child kernel from device
        child_kernel<<<n/256, 256>>>(data, n);
        cudaDeviceSynchronize(); // Wait for children
    }
}
```

### CDP Limitations
- Max nesting depth: 24 levels (theoretical), practical limit ~2-3
- No CUDA Graph integration for child launches
- Higher overhead than host-side launches for regular patterns
- Debugging more complex (use Compute Sanitizer with `--tool synccheck`)

---

## 10. CUDA Graphs & Task Graph Acceleration

### CUDA Graphs (Introduced CUDA 10, Enhanced in 11)

**Concept**: Define-once, run-repeatedly execution model. A graph captures kernel launches, memory copies, and dependencies into a reusable structure, eliminating per-launch CPU overhead.

### Ampere Hardware Acceleration
A100 adds dedicated hardware to:
- **Prefetch grid launch descriptors, instructions, and constants** between graph nodes
- Faster inter-kernel transitions (no CPU round-trip)
- Reduced launch latency: ~3-5μs per kernel → <1μs in graph chains

### Key CUDA 11 Graph Features

**Graph Creation (Explicit API)**:
```cpp
cudaGraph_t graph;
cudaGraphCreate(&graph, 0);

cudaKernelNodeParams kernelParams = {0};
kernelParams.func = (void*)myKernel;
kernelParams.gridDim = dim3(N/256);
kernelParams.blockDim = dim3(256);
kernelParams.kernelParams = args;

cudaGraphNode_t node;
cudaGraphAddKernelNode(&node, graph, NULL, 0, &kernelParams);
```

**Stream Capture (More Convenient)**:
```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);

cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

// Queue operations as usual
kernel_A<<<grid, block, 0, stream>>>(args_A);
kernel_B<<<grid, block, 0, stream>>>(args_B);
cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream);

cudaStreamEndCapture(stream, &graph);

// Instantiate and launch
cudaGraphExec_t instance;
cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
cudaGraphLaunch(instance, stream);
```

**In-Place Graph Updates** (CUDA 11):
```cpp
// Update parameters without rebuilding graph
cudaGraphExecKernelNodeSetParams(instance, node, &newKernelParams);
// Much faster than re-instantiation
```

**Cooperative Kernel in Graphs** (CUDA 11):
```cpp
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
void* args[] = {&d_data};
cudaLaunchCooperativeKernel((void*)kernel, grid, block, args);
cudaStreamEndCapture(stream, &graph);
```

---

## 11. Unified Memory Improvements

### CUDA 11/Ampere Enhancements

**1. Hardware-Accelerated Page Migration on A100**:
- Page fault handling is faster with improved TLB (Translation Lookaside Buffer)
- Multi-Instance GPU support: each MIG instance has isolated address translation
- Direct peer access via NVLink with coherent page tables

**2. `cudaMemPrefetchAsync()` improvements**:
```cpp
// Prefetch to GPU 0
cudaMemPrefetchAsync(data, size, 0, stream);
// Prefetch to CPU (useful after GPU computation)
cudaMemPrefetchAsync(data, size, cudaCpuDeviceId, stream);
```

**3. Memory Advise Hints**:
```cpp
cudaMemAdvise(data, size, cudaMemAdviseSetReadMostly, deviceId);   // Data rarely written
cudaMemAdvise(data, size, cudaMemAdviseSetPreferredLocation, deviceId); // Keep here
cudaMemAdvise(data, size, cudaMemAdviseSetAccessedBy, deviceId);   // Enable direct access
```

**4. System-wide Unified Memory (ATS - Address Translation Services)**:
- With PCIe Gen 4 + ATS-capable CPU: full coherent unified address space
- Enables oversubscription beyond GPU memory with reasonable performance
- Multi-GPU peer access with unified addressing

**5. `cudaMallocManaged()` with `cudaMemAttachGlobal`**:
```cpp
float* data;
cudaMallocManaged(&data, size, cudaMemAttachGlobal);
// Accessible from any GPU in the system
```

---

## 12. Stream-Ordered Memory Allocator

### Problem Solved
Traditional `cudaMalloc`/`cudaFree` are synchronizing and expensive. Multiple threads allocating/freeing causes serialization. No ordering relative to stream work.

### Solution: Stream-Ordered Allocation (CUDA 11.2+)

```cpp
cudaMemPool_t pool;
cudaDeviceGetDefaultMemPool(&pool, 0);

// Set pool size limit (optional)
size_t maxPoolSize = 2ULL * 1024 * 1024 * 1024; // 2 GB
cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &maxPoolSize);

// Allocate from pool (async, stream-ordered)
float* data;
cudaMallocAsync(&data, sizeof(float) * N, stream);

// Use data in kernels on same stream
myKernel<<<grid, block, 0, stream>>>(data);

// Free (also async, stream-ordered)
cudaFreeAsync(data, stream);
```

### Benefits
- **No synchronization** on allocation/free
- **Thread-safe** by default (pool-based)
- **Stream ordering**: allocation becomes available exactly when prior work completes
- **Reuses memory** without returning to OS
- **Inter-stream dependencies** handled automatically
- **Works with CUDA Graphs**

### Memory Pools
```cpp
// Create custom pool
cudaMemPoolProps poolProps = {};
poolProps.allocType = cudaMemAllocationTypePinned;
poolProps.location.id = 0;
poolProps.location.type = cudaMemLocationTypeDevice;

cudaMemPool_t customPool;
cudaMemPoolCreate(&customPool, &poolProps);

// Trim pool to release unused memory back to OS
cudaMemPoolTrimTo(customPool, minBytesToKeep);
```

---

## 13. Asynchronous Operations & Barriers

### 13.1 Asynchronous Copy (async-copy)

**Problem**: Global→Shared memory copy goes through registers (global→register→shared), wasting register bandwidth.

**Solution**: Hardware-accelerated async copy on A100 (sm_80):
```cpp
__shared__ float smem[BLOCK_SIZE];

// Traditional (through register):
// float tmp = global[i]; smem[i] = tmp;

// Async copy (hardware-accelerated on A100):
cuda::memcpy_async(smem + offset, global + offset, sizeof(float) * count, pipeline);
pipeline.commit();
pipeline.wait_prior<0>(); // Wait for all but last stage
```

**Benefits**:
- Bypasses register file entirely
- Overlaps copy with computation
- Reduces register pressure → higher occupancy
- Hardware path: global memory → L1/cache → shared memory directly

### 13.2 Asynchronous Barriers

**Ampere introduces hardware barriers in shared memory**:
```cpp
#include <cuda/barrier>
cuda::barrier<cuda::thread_scope_block> barrier;

// Phase 1: Issue async copies
cuda::memcpy_async(smem, gmem, size, barrier);

// Phase 2: Do computation on previously loaded data
compute_phase(smem_prev);

// Phase 3: Arrive and wait for new data
barrier.arrive_and_wait();

// Phase 4: Compute on newly arrived data
compute_phase(smem_current);
```

**Key properties**:
- Split arrive/wait operations for producer-consumer patterns
- Hardware-accelerated on A100 (shared memory barrier unit)
- Supports warp-level and block-level synchronization
- More flexible than `__syncthreads()` — can arrive without waiting

### 13.3 Pipeline Stages (Multi-Stage Pipelining)
```cpp
constexpr int NUM_STAGES = 3;
cuda::pipeline<cuda::thread_scope_block> pipeline = cuda::make_pipeline();

for (int stage = 0; stage < NUM_STAGES; stage++) {
    // Issue async copy for this stage
    pipeline.producer_acquire();
    cuda::memcpy_async(&smem[stage % NUM_STAGES], &gmem[stage * CHUNK], CHUNK, pipeline);
    pipeline.producer_commit();

    if (stage >= NUM_STAGES - 1) {
        pipeline.consumer_wait();
        // Compute on data from (stage - NUM_STAGES + 1)
        compute(&smem[(stage - NUM_STAGES + 1) % NUM_STAGES]);
        pipeline.consumer_release();
    }
}
```

---

## 14. L2 Cache Management & Residency Controls

### Use Cases
- **Persisting data**: Ping-pong buffers, LSTM recurrent weights, lookup tables
- **Streaming data**: One-pass sequential accesses that shouldn't evict persistent data

### API
```cpp
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);

// Reserve 50% of L2 for persistent accesses
size_t persistSize = min((size_t)(prop.l2CacheSize * 0.50),
                         prop.persistingL2CacheMaxSize);
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, persistSize);

// Set access policy on a stream
cudaStreamAttrValue attr;
attr.accessPolicyWindow.base_ptr  = persistentData;
attr.accessPolicyWindow.num_bytes = persistentDataSize;
attr.accessPolicyWindow.hitRatio  = 1.0f; // All data fits
attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;

cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);

// Also works in CUDA Graph kernel nodes:
cudaStreamSetAttribute(stream, cudaStreamNodeAttributeAccessPolicyWindow, &attr);
```

### Performance Impact
- Can provide 2-10× speedup for working-set-fits-in-L2 workloads
- Eliminates redundant DRAM traffic for frequently accessed data
- Critical for latency-sensitive inference workloads

---

## 15. nvcc Compilation Flags for CUDA 11.x

### Architecture Specification
```bash
# Target A100 specifically
nvcc -arch=sm_80 ...           # Ampere (A100, A30, A40)
nvcc -arch=compute_80 ...      # PTX only for sm_80
nvcc -arch=sm_80 -code=sm_80   # SASS for sm_80 only
nvcc -gencode arch=compute_80,code=sm_80   # Common form

# Multi-architecture fat binary
nvcc -gencode arch=compute_70,code=sm_70 \  # Volta (V100)
     -gencode arch=compute_75,code=sm_75 \  # Turing (T4)
     -gencode arch=compute_80,code=sm_80 \  # Ampere (A100)
     -gencode arch=compute_80,code=compute_80 \  # Forward-compat PTX
     ...

# Ampere also has sm_86 (GA106/GA102 - RTX 30xx consumer)
nvcc -arch=sm_86 ...
```

### C++ Standard
```bash
nvcc -std=c++17 ...            # C++17 support (new in CUDA 11)
nvcc -std=c++14 ...            # Default in some CUDA 11 versions
```

### Link-Time Optimization
```bash
nvcc -dlink-time-opt ...       # LTO across compilation units (new in CUDA 11)
# Stores intermediate code, performs cross-file inlining at link time
```

### Optimization Flags
```bash
-O3                            # Host-side optimization
--ptxas-options=-v             # Print register/shared memory usage
--ptxas-options=-O3            # Device-side optimization
-maxrregcount=64               # Limit registers per thread (occupancy control)
--use_fast_math                # Use fast (less precise) math intrinsics
--extra-device-vectorization   # Enable more aggressive vectorization
```

### Debug & Profiling
```bash
-G                             # Device debug info (disables optimizations)
-lineinfo                       # Line number info (no optimization impact)
# For profiling with Nsight
# (no special flags, but don't use -G for profiling)
```

### Host Compiler Passthrough (New in CUDA 11)
```bash
nvcc -ccbin g++-11 ...         # Specify host compiler
nvcc --allow-unsupported-compiler  # Allow unlisted host compilers
```

### Tensor Core & Architecture Specific
```bash
nvcc -arch=sm_80 \
     --use_fast_math \          # Enables __fmul_rn, etc.
     -t=8 \                     # Threads per block hint
     -dlcm=cg                   # Cache global loads in L2 only
```

### Linking Libraries
```bash
nvcc -lcublas -lcusparse -lcufft -lcusolver -lcurand -lnvToolsExt ...
```

---

## 16. C++17 Device Features

### CUDA 11.x C++17 Support

**nvcc C++17 features available on device code**:

1. **`if constexpr`** — compile-time branching:
   ```cpp
   template<typename T>
   __device__ void process(T val) {
       if constexpr (std::is_same_v<T, float>) {
           // Float-specific device code
           val = __expf(val);
       } else {
           // Generic path
       }
   }
   ```

2. **Structured bindings** (host + device):
   ```cpp
   __device__ auto getPair() { return thrust::make_pair(1.0f, 2.0f); }
   auto [x, y] = getPair(); // On device too
   ```

3. **Fold expressions**:
   ```cpp
   template<typename... Args>
   __device__ auto sum(Args... args) {
       return (args + ...);
   }
   ```

4. **`constexpr` lambdas**:
   ```cpp
   constexpr auto square = [](auto x) { return x * x; };
   __device__ float result = square(3.0f); // Computed at compile time
   ```

5. **`std::optional`, `std::variant`** — host-side only (not in device code)

6. **Inline variables**:
   ```cpp
   __device__ inline constexpr int BLOCK_SIZE = 256;
   ```

7. **Nested namespaces**:
   ```cpp
   namespace mylib::cuda::kernels {
       __global__ void myKernel() { ... }
   }
   ```

8. **Class template argument deduction (CTAD)**:
   ```cpp
   // Host-side template argument deduction
   thrust::pair p(1.0f, 2); // Deduces <float, int>
   ```

### Important Notes
- `std::` library features (optional, variant, any, string_view) are **host-only**
- Device-side C++17 features are language constructs, not library features
- Always check with `-std=c++17` flag and `__cplusplus >= 201703L`

---

## 17. CUDA 11.x Minor Version Features

### CUDA 11.0 (May 2020) — A100 Launch
- Ampere architecture support (sm_80)
- All features described in this document
- TF32, BF16, FP64 Tensor Cores
- Multi-Instance GPU
- Async copy, barriers, L2 residency
- CUB integrated into toolkit
- Compute Sanitizer (replaces memcheck)

### CUDA 11.1 (October 2020)
- A100-80GB support
- Ampere for GeForce RTX 30 series (sm_86)
- New WMMA matrix shapes for sm_86

### CUDA 11.2 (December 2020)
- **Stream-ordered memory allocator** (`cudaMallocAsync`/`cudaFreeAsync`)
- **Dependency relaxed memory access** (for `__ldg` etc.)
- Improved Nsight Compute profiler
- Enhanced occupancy calculator
- Improved CUDA Graph update API

### CUDA 11.3 (April 2021)
- **Device-side `memcpy_async`** improvements
- Cooperative groups enhancements
- cuBLAS performance improvements for A100
- Nsight Systems improvements

### CUDA 11.4 (June 2021)
- **Graph memory nodes** — `cudaMallocAsync`/`cudaFreeAsync` inside graphs
- Enhanced graph APIs
- Multi-process service (MPS) improvements
- cuBLAS GEMM algorithm selection improvements

### CUDA 11.5 (November 2021)
- **CUDA Enhanced Compatibility** — new minor version driver supports older toolkit
- Improved compilation times
- Nsight Developer Tools enhancements

### CUDA 11.6 (January 2022)
- **CUDA Graph conditional nodes** (experimental)
- Additional math operations
- cuFFT improvements for A100

### CUDA 11.7 (May 2022)
- Grace Hopper architecture support
- Cooperative groups `invoke()` on device
- Nsight Compute 2022.2

### CUDA 11.8 (October 2022)
- Hopper architecture support (sm_90)
- FP8 data type support preparation
- L2 residency improvements
- Last CUDA 11.x release

---

## 18. Performance Specifications

### A100-80GB SXM Performance Summary
| Metric | Value |
|--------|-------|
| FP64 (CUDA) | 9.7 TFLOPS |
| FP32 (CUDA) | 19.5 TFLOPS |
| FP64 Tensor | 19.5 TFLOPS |
| TF32 Tensor | 156 / 312 TFLOPS* |
| BF16 Tensor | 312 / 624 TFLOPS* |
| FP16 Tensor | 312 / 624 TFLOPS* |
| INT8 Tensor | 624 / 1,248 TOPS* |
| INT4 Tensor | 1,248 / 2,496 TOPS* |
| Memory | 80 GB HBM2e |
| Memory BW | 2,039 GB/s |
| L2 Cache | 40 MB |
| L1+Shared/SM | 192 KB |
| NVLink BW | 600 GB/s |
| PCIe | Gen 4 ×16 (31.5 GB/s) |
| TDP | 400W |

*\*With 2:4 structured sparsity*

---

## 19. Implications for Our ML Trainer

### For `ultimate-ai-model` PyTorch/CUDA/Triton Project

1. **TF32 is default** — PyTorch on Ampere uses TF32 for FP32 matmul automatically. Good for speed, but if you need full FP32 precision:
   ```python
   torch.backends.cuda.matmul.allow_tf32 = False
   torch.backends.cudnn.allow_tf32 = False
   ```

2. **BFloat16 training** — Preferred over FP16 for Ampere:
   - No loss scaling needed (same range as FP32)
   - Same throughput as FP16 on Tensor Cores
   - More stable training for large models

3. **Triton kernel optimization for sm_80**:
   ```python
   # In Triton, target A100 specifically
   @triton.autotune(
       configs=[
           triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE': 8}, num_stages=3, num_warps=8),
       ],
       key=['M', 'N', 'K'],
   )
   @triton.jit
   def matmul_kernel(...):
   ```

4. **Stream-ordered allocation** — Use `torch.cuda.memory.CUDAPluggableAllocator` or `torch.cuda.memory._host_allocator()` for stream-ordered memory management.

5. **CUDA Graphs for training** — Wrap repetitive training loops:
   ```python
   # PyTorch CUDA Graphs
   g = torch.cuda.CUDAGraph()
   with torch.cuda.graph(g):
       static_output = model(static_input)

   # Replay for each iteration
   static_input.copy_(new_data)
   g.replay()
   output = static_output
   ```

6. **MIG for multi-tenant inference** — Partition A100 for serving multiple models at guaranteed latency.

7. **Async memory operations** — Use `memory_format=torch.channels_last` for optimal cache behavior on Ampere. The 40MB L2 cache is huge — design algorithms to fit working sets in L2 when possible.

8. **Sparsity for inference** — Train with 2:4 structured sparsity for 2× inference throughput on A100:
   ```python
   from torch.nn.utils import prune
   # Apply 2:4 structured pruning
   prune.ln_structured(module, 'weight', amount=0.5, n=2, dim=0)
   ```

---

## Key References

1. **NVIDIA Ampere Architecture In-Depth** — NVIDIA Technical Blog (developer.nvidia.com)
2. **CUDA 11 Features Revealed** — NVIDIA Technical Blog
3. **NVIDIA A100 Tensor Core GPU Architecture Whitepaper** — NVIDIA
4. **CUDA C++ Programming Guide** — docs.nvidia.com/cuda/cuda-c-programming-guide/
5. **CUTLASS Documentation** — github.com/NVIDIA/cutlass
6. **CUDA Toolkit Documentation** — docs.nvidia.com/cuda/
