# CUDA 12.x (Hopper) — Comprehensive Research Summary

> **Date**: 2026-07-25
> **Status**: Research complete
> **Scope**: CUDA 12.0–12.9, Hopper architecture (sm_90, H100)
> **Relevance**: Ultimate AI Model project — BitNet b1.58 × SubQSA trainer targeting Hopper tensor cores, FP8, TMA, and distributed shared memory

---

## Table of Contents

1. [Architecture Overview: Hopper (sm_90)](#1-architecture-overview-hopper-sm_90)
2. [H100 GPU Specifications](#2-h100-gpu-specifications)
3. [Hopper-Specific Features](#3-hopper-specific-features)
4. [Tensor Memory Accelerator (TMA)](#4-tensor-memory-accelerator-tma)
5. [Thread Block Clusters](#5-thread-block-clusters)
6. [Distributed Shared Memory (DSMEM)](#6-distributed-shared-memory-dsmem)
7. [Asynchronous Transaction Barrier](#7-asynchronous-transaction-barrier)
8. [FP8 Data Format & Tensor Cores](#8-fp8-data-format--tensor-cores)
9. [Transformer Engine Integration](#9-transformer-engine-integration)
10. [CUDA Graphs Enhancements](#10-cuda-graphs-enhancements)
11. [Programmatic Dependent Launch (PDL)](#11-programmatic-dependent-launch-pdl)
12. [C++20 Device Features](#12-c20-device-features)
13. [Dynamic Parallelism v2](#13-dynamic-parallelism-v2)
14. [Lazy Module Loading](#14-lazy-module-loading)
15. [nvJitLink: JIT Link-Time Optimization](#15-nvjitlink-jit-link-time-optimization)
16. [CUDA Minimal API (Driver API)](#16-cuda-minimal-api-driver-api)
17. [Memory Synchronization Domains](#17-memory-synchronization-domains)
18. [Reduced Register Usage & Occupancy](#18-reduced-register-usage--occupancy)
19. [nvcc Flags for CUDA 12.x](#19-nvcc-flags-for-cuda-12x)
20. [Compilation & Multi-Architecture Support](#20-compilation--multi-architecture-support)
21. [Relevance to Ultimate AI Model](#21-relevance-to-ultimate-ai-model)
22. [References](#22-references)

---

## 1. Architecture Overview: Hopper (sm_90)

The NVIDIA Hopper GPU architecture (compute capability 9.0) is NVIDIA's first "truly asynchronous GPU," designed to maximize data movement overlap with computation. It represents the most significant architectural leap since Volta, adding an entirely new level to the CUDA programming hierarchy.

### Programming Hierarchy (New in Hopper)

| Level | Description | Synchronization |
|-------|-------------|-----------------|
| **Grid** | All blocks in a kernel launch | No ordering guarantees |
| **Thread Block Cluster** ⭐ NEW | Group of blocks guaranteed concurrent on nearby SMs | `__cluster_sync()`, DSMEM, hardware barriers |
| **Thread Block** | Threads on one SM | `__syncthreads()` |
| **Warp** | 32 threads, SIMT | `__syncwarp()` |
| **Thread** | Single execution context | N/A |

The cluster level is the key Hopper innovation — it enables programmatic control of locality at a granularity larger than a single thread block on a single SM.

### Key Architectural Principles

1. **Asynchronous everything**: TMA, async transaction barriers, async pipeline, end-to-end async data movement
2. **Cluster-level cooperation**: Hardware-accelerated multi-block coordination within a GPC
3. **FP8 native support**: 2× throughput over FP16, half the memory footprint
4. **Transformer Engine**: Software+hardware co-design for dynamic precision management

---

## 2. H100 GPU Specifications

### Hardware Specs

| Specification | H100 SXM5 | H100 PCIe | A100 (comparison) |
|---------------|-----------|-----------|-------------------|
| **Architecture** | Hopper | Hopper | Ampere |
| **Process** | TSMC 4N (custom) | TSMC 4N (custom) | TSMC 7nm N7 |
| **Transistors** | 80 billion | 80 billion | 54.2 billion |
| **Die Size** | 814 mm² | 814 mm² | 826 mm² |
| **SMs** | 132 | 114 | 108 |
| **FP32 Cores / SM** | 128 | 128 | 64 |
| **FP32 Cores / GPU** | 16,896 | 14,592 | 6,912 |
| **Tensor Cores / SM** | 4 | 4 | 4 |
| **Tensor Cores / GPU** | 528 | 456 | 432 |
| **Memory** | 80 GB HBM3 | 80 GB HBM2e | 40 GB HBM2 |
| **Memory Bandwidth** | 3 TB/s | 2 TB/s | 1.55 TB/s |
| **L2 Cache** | 50 MB | 50 MB | 40 MB |
| **Shared Memory / SM** | Up to 228 KB | Up to 228 KB | Up to 164 KB |
| **Register File / SM** | 256 KB (64K regs) | 256 KB (64K regs) | 256 KB (64K regs) |
| **Max Warps / SM** | 64 | 64 | 64 |
| **NVLink** | 4th gen, 900 GB/s | 4th gen | 3rd gen, 600 GB/s |
| **PCIe** | Gen 5 (128 GB/s) | Gen 5 | Gen 4 (64 GB/s) |
| **TDP** | 700 W | 350 W | 400 W |

### Compute Performance (TFLOPS)

| Data Type | H100 SXM5 Dense | H100 SXM5 Sparse | A100 | Speedup |
|-----------|-----------------|-------------------|------|---------|
| **FP8 Tensor** | 2,000 | 4,000 | N/A | 6.4× vs A100 FP16 |
| **FP16 Tensor** | 1,000 | 2,000 | 312 | 3.2× |
| **BF16 Tensor** | 1,000 | 2,000 | 312 | 3.2× |
| **TF32 Tensor** | 500 | 1,000 | 156 | 3.2× |
| **FP64 Tensor** | 60 | N/A | 19.5 | 3.1× |
| **FP32 (non-Tensor)** | 60 | N/A | 19.5 | 3.1× |
| **FP64 (non-Tensor)** | 30 | N/A | 9.7 | 3.1× |
| **INT8 Tensor** | 2,000 TOPS | 4,000 TOPS | 624 TOPS | 3.2× |

### Key H100 SM Characteristics

- **2× FP32 cores per SM**: 128 vs 64 in A100
- **2× FP32 throughput per SM**: Clock-for-clock, H100 SM has 2× more FP32 ops/cycle
- **256 KB combined shared memory + L1**: 33% larger than A100's 192 KB
- **228 KB max shared memory per SM** (up from 164 KB)
- **227 KB max shared memory per thread block** (1 KB reserved per block)

---

## 3. Hopper-Specific Features

### Summary of New Hardware Features

| Feature | Description | Benefit |
|---------|-------------|---------|
| **TMA** | Tensor Memory Accelerator — hardware async copy engine | Frees threads from data movement |
| **Thread Block Clusters** | New hierarchy level above thread block | Cross-SM cooperation |
| **DSMEM** | Distributed Shared Memory | Direct SM-to-SM data exchange |
| **Async Transaction Barriers** | Count transactions, not just thread arrivals | Better async pipeline control |
| **FP8 Tensor Cores** | E4M3 and E5M2 native support | 2× throughput, half memory |
| **Transformer Engine** | Dynamic FP8↔FP16 per-layer | Automated mixed precision |
| **WGMMA** | Warp Group MMA instructions | Higher tensor core utilization |
| **DPX** | Dynamic programming instructions | 7× speedup for DP algorithms |
| **4th Gen NVLink** | 900 GB/s, 18 links | 3× all-reduce improvement |
| **Inline Compression** | Hardware memory compression | Effective bandwidth increase |
| **Memory Sync Domains** | Finer-grained ordering control | Reduced unnecessary fencing |
| **Cluster Launch Control** | Work stealing across clusters | Better load balancing |

---

## 4. Tensor Memory Accelerator (TMA)

### What It Is

TMA is a **hardware asynchronous copy engine** in each Hopper SM that can transfer 1D to 5D tensors between:
- Global memory ↔ Shared memory
- Shared memory of one SM → Shared memory of another SM (within a cluster)

### How It Differs from A100

| Aspect | A100 (LDGSTS) | H100 (TMA) |
|--------|---------------|------------|
| **Address generation** | Threads compute all addresses | Single thread issues command |
| **Loop management** | Threads loop over copy region | Hardware unit handles iteration |
| **Dimensionality** | 1D (linear) | 1D to 5D tensors |
| **Register usage** | Uses registers for data movement | Zero register usage for copies |
| **Warp specialization** | Not practical | Producer warps issue TMA, consumer warps compute |
| **Reduction ops** | Not supported | Element-wise reduction on write (add/min/max, and/or) |

### API Exposure

```cpp
#include <cuda/barrier>
#include <cuda/pipeline>

// Single thread issues the copy
cuda::memcpy_async(dst_shared, src_global, size, barrier);
// All threads can wait on the barrier
barrier.arrive_and_wait();
```

### Key Advantages

1. **Frees threads**: A single thread can issue large tensor transfers. The entire block continues working while data is in flight.
2. **Warp specialization**: Producer warps specialize in data movement; consumer warps specialize in computation.
3. **Multi-dimensional**: Native support for 2D/3D/4D/5D tensor transfers with strides — no manual index calculation.
4. **Element-wise reductions**: Writes from SMEM → global can perform atomic reductions (add, min, max, and, or) on most data types.

### TMA for This Project

For the BitNet b1.58 × SubQSA trainer:
- **Ternary matmul**: TMA can asynchronously load packed weight tiles and activation tiles into shared memory without consuming registers.
- **SubQSA compression branch**: The unfold operation (`k.unfold()`) that materializes large intermediates could be replaced with TMA-driven multi-dimensional copies of K/V blocks.
- **Selection branch**: TMA's 5D tensor support maps directly to the `(batch, head, block, seq, dim)` structure of attention tensors.

---

## 5. Thread Block Clusters

### What They Are

A **thread block cluster** is a group of thread blocks (up to 8 portably, up to 16 on H100 with opt-in) that are:
- **Guaranteed concurrently scheduled** on physically nearby SMs within a GPC
- **Connected by a dedicated SM-to-SM network** for fast data exchange
- **Able to access each other's shared memory** (Distributed Shared Memory)

### CUDA API

```cpp
// Cooperative groups API
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void cluster_kernel() {
    auto cluster = cg::this_cluster();
    unsigned cluster_size = cluster.dim_threads(); // total threads in cluster
    unsigned block_rank = cluster.block_rank();    // which block in cluster
    
    // Sync all threads across the cluster
    cluster.sync();
    
    // Get pointer to another block's shared memory
    void* remote_smem = cluster.map_shared_rank(other_block_rank, local_smem_ptr);
}
```

### Cluster Size

| Setting | Cluster Size | Notes |
|---------|-------------|-------|
| **Portable** | Up to 8 blocks | Works on all Hopper GPUs |
| **Non-portable (H100)** | Up to 16 blocks | Requires `cudaFuncAttributeNonPortableClusterSizeAllowed` |

### Performance Considerations

- Larger cluster sizes may reduce the maximum number of active blocks across the GPU.
- Use `cudaOccupancyMaxActiveClusters` to compute optimal cluster launch configuration.
- Clusters operate within a GPC (GPU Processing Cluster) — SMs are physically close.

### Cluster Launch

```cpp
// Launch with cluster size
cudaLaunchCooperativeKernel(
    kernel, 
    grid_dim, block_dim, 
    args,
    0,        // shared memory
    stream,
    cluster_dim  // cluster dimensions
);
```

### Relevance to This Project

Thread block clusters directly benefit the SubQSA attention pattern:
- **Multi-block cooperative attention**: Multiple thread blocks can cooperatively compute different branches (compression, selection, sliding window) and exchange results via DSMEM.
- **Ternary matmul tiling**: Clusters enable larger tile sizes by distributing K-dimension tiles across SMs with hardware-accelerated communication.
- **Fused kernels**: The `subqsa_combine_kernel` (gate → blend → norm → O projection) can span multiple cluster blocks for parallelism.

---

## 6. Distributed Shared Memory (DSMEM)

### What It Is

DSMEM allows **direct load, store, and atomic operations** on the shared memory of other thread blocks within the same cluster. The shared memory segments from all blocks in a cluster are mapped into each thread's generic address space.

### Key Characteristics

| Aspect | Detail |
|--------|--------|
| **Scope** | Within a thread block cluster only |
| **Addressing** | Generic pointers via `cooperative_groups` API |
| **Operations** | Load, store, atomics (add, min, max, and, or, xor) |
| **Bandwidth** | ~7× faster than global memory for inter-block exchange |
| **Latency** | Similar to L2 cache access |
| **L2 coexistence** | DSMEM and L2 can be used simultaneously |

### API

```cpp
__shared__ float smem[1024];

auto cluster = cg::this_cluster();

// Get pointer to remote block's shared memory
float* remote = (float*)cluster.map_shared_rank(remote_block_rank, smem);

// Direct load/store to remote SMEM
float val = remote[threadIdx.x];   // Load from remote
remote[threadIdx.x] = new_val;     // Store to remote
```

### Data Exchange Pattern

```
A100 (without clusters):
  Block 0 SMEM → Global Memory → Block 1 SMEM  (2 memory hops)
  
H100 (with DSMEM):
  Block 0 SMEM → SM-to-SM Network → Block 1 SMEM  (direct, ~7× faster)
```

### Relevance to This Project

- **Compression → Selection pipeline**: The compression branch produces compressed K/V blocks that the selection branch needs. Instead of writing to global memory, the compression block can write directly to the selection block's SMEM via DSMEM.
- **Fused attention branches**: All three SubQSA branches (compression, selection, sliding window) could run as separate cluster blocks and exchange their outputs via DSMEM before the gating/blend step.
- **Ternary matmul tiling**: For large matrix dimensions, DSMEM allows distributing K-dimension tiles across cluster blocks for cooperative computation.

---

## 7. Asynchronous Transaction Barrier

### A100 Async Barriers (Baseline)

A100 introduced split barriers that separate "arrive" from "wait":
```cpp
cuda::barrier bar;
// Producer: arrive when data is ready
bar.arrive();
// Consumer: wait for all producers
bar.arrive_and_wait();
// Advantage: early arrivers can do other work while waiting
```

### H100 Async Transaction Barriers (New)

H100 extends the concept with **transaction counting** — barriers that track not just thread arrivals, but also the number of data bytes written to shared memory.

```cpp
// A single thread issues data + transaction count to SMEM
// The barrier tracks both thread arrivals AND byte counts
// Threads block at wait() until:
//   1. All producer threads have arrived AND
//   2. Total transaction bytes reach expected value
```

### Key Differences

| Aspect | Async Barrier (A100) | Async Transaction Barrier (H100) |
|--------|---------------------|----------------------------------|
| **Counts** | Thread arrivals only | Thread arrivals + data transactions |
| **Wait behavior** | Spins on shared memory | Threads can **sleep** until arrival |
| **Producer model** | All threads produce equally | Single thread can produce, others consume |
| **TMA integration** | Manual coordination | Native TMA completion tracking |
| **Use case** | General sync | Data pipeline with variable data sizes |

### How It Works

1. Producer thread(s) write data to shared memory with an associated transaction (byte) count
2. The barrier accumulates transaction counts from all producers
3. Consumer threads at `wait()` block until:
   - All expected producers have called `arrive()`
   - Sum of transaction counts ≥ expected byte count

### Relevance to This Project

- **TMA + barrier pipeline**: TMA copies are tracked by transaction barriers, enabling fully asynchronous data movement with precise completion tracking.
- **Fused kernel coordination**: In the `subqsa_combine_kernel`, different warps/threads produce partial results (gate values, branch outputs) that need to be combined. Transaction barriers ensure all data is ready before the blend step.
- **Warp specialization**: Producer warps issue TMA copies and arrive with transaction counts; consumer warps process already-available data and wait only when needed.

---

## 8. FP8 Data Format & Tensor Cores

### Two FP8 Formats

| Format | Sign | Exponent | Mantissa | Range | Use Case |
|--------|------|----------|----------|-------|----------|
| **E4M3 (FP8)** | 1 | 4 | 3 | ±448 | Forward pass (better precision) |
| **E5M2** | 1 | 5 | 2 | ±57344 | Backward pass (larger range) |

### FP8 Tensor Core Performance

| Metric | FP16 Tensor Core | FP8 Tensor Core | Improvement |
|--------|-----------------|-----------------|-------------|
| **Throughput per SM** | 1,000 TFLOPS | 2,000 TFLOPS | 2× |
| **With sparsity** | 2,000 TFLOPS | 4,000 TFLOPS | 2× |
| **Memory footprint** | 16 bits/element | 8 bits/element | 2× reduction |
| **vs A100 FP16** | — | 6.4× total | 6.4× |

### FP8 Tensor Core Capabilities

- **Accumulators**: FP32 and FP16 supported
- **Input types**: FP8 (E4M3), FP8 (E5M2), FP16, BF16, TF32, FP64, INT8
- **WGMMA instructions**: Warp Group MMA — a group of 4 warps cooperatively executes a matrix multiply
- **Structured sparsity**: 2:4 structured sparsity doubles throughput

### WGMMA (Warp Group Matrix Multiply-Accumulate)

New in Hopper, WGMMA instructions allow a **group of 4 warps** (128 threads) to cooperatively execute a matrix multiply operation on the tensor cores. This is more efficient than the per-warp WMMA approach used in Ampere/Turing.

```ptx
// PTX WGMMA instruction (conceptual)
wgmma.mma_async.aligned.m64n256k32.f32.e4m3.e4m3  d, a, b, c;
// 4 warps cooperate, M=64, N=256, K=32
// Inputs: E4M3 (FP8), Accumulator: FP32
```

### Relevance to This Project

- **Ternary weight quantization**: While ternary weights are {-1, 0, +1} (2 bits), the FP8 tensor cores could be used for the **activation quantization path** — currently activations are INT8 (8-bit signed), but FP8 E4M3 could provide better gradient flow.
- **Attention computation**: The softmax and attention score computation could benefit from FP8 for the forward pass (with FP16 for backward).
- **Compression MLP**: The learned compression function φ (MLP) could run in FP8 for faster inference and faster compression branch computation.
- **WGMMA for larger tiles**: The 4-warp cooperative MMA maps well to the SubQSA's multi-head attention pattern where each head processes independent tiles.

---

## 9. Transformer Engine Integration

### What It Is

The Transformer Engine (TE) is a **software + hardware co-design** that:
1. Analyzes output statistics of each transformer layer
2. Dynamically chooses between FP8 and FP16 precision per layer
3. Automatically handles re-casting and scaling between FP8 and FP16
4. Maintains model accuracy while maximizing performance

### How It Works

```
For each transformer layer:
  1. Run forward pass in FP8
  2. Analyze output tensor statistics (histogram of values)
  3. Compute optimal scaling factor for next iteration
  4. If accuracy is degrading → switch to FP16 for this layer
  5. If accuracy is fine → stay in FP8
  6. Store per-layer scaling factors for next iteration
```

### Performance Claims

| Operation | Speedup vs A100 |
|-----------|----------------|
| **AI Training (LLMs)** | Up to 9× faster |
| **AI Inference (LLMs)** | Up to 30× faster |

### Integration with PyTorch

The Transformer Engine is available as a standalone library (`transformer_engine`) that provides:
- `te.Linear` — FP8-capable linear layer
- `te.LayerNorm` — Layer normalization with FP8 support
- `te.TransformerLayer` — Full transformer layer with TE
- `te.MultiheadAttention` — Attention with FP8

```python
import transformer_engine.pytorch as te

# Drop-in replacement for nn.Linear with FP8
linear = te.Linear(in_features, out_features)
output = linear(input)  # Automatically uses FP8 when beneficial
```

### Relevance to This Project

The Transformer Engine is **directly relevant** to the BitNet b1.58 × SubQSA project:

1. **Compression MLP**: The φ function (2-layer MLP: Linear→SiLU→Linear) could use `te.Linear` for FP8 acceleration.
2. **Gate MLP**: The gate computation (Linear→SiLU→Linear→sigmoid) could benefit from TE.
3. **BitLinear integration**: TE's dynamic precision management could complement the ternary quantization by handling the FP portions of the computation.
4. **SubQSA routing projections**: The FP routing projection (`routing_k_proj`) used for attention score routing could run through TE.

**Challenge**: TE is designed for standard transformer layers. Integrating it with the custom BitLinear + SubQSA architecture requires careful handling of the quantization boundaries.

---

## 10. CUDA Graphs Enhancements

### CUDA Graphs Fundamentals

CUDA Graphs capture a sequence of GPU operations as a reusable graph, eliminating per-launch CPU overhead:

```cpp
// Capture
cudaGraph_t graph;
cudaStreamBeginCapture(stream);
    kernel_a<<<grid, block, 0, stream>>>(args...);
    kernel_b<<<grid, block, 0, stream>>>(args...);
    cudaMemcpyAsync(...);
cudaStreamEndCapture(stream, &graph);

// Instantiate and replay
cudaGraphExec_t instance;
cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
cudaGraphLaunch(instance, stream);  // Zero CPU overhead on replay
```

### CUDA 12.x Enhancements

| Feature | Description |
|---------|-------------|
| **Device-side graph launch** | Launch graphs from device code via `cudaLaunchGraph()` |
| **Graph update API** | Update kernel parameters without full reinstantiation |
| **Conditional nodes** | If/while/switch nodes for conditional execution within graphs |
| **Memory allocation nodes** | cudaMallocAsync/free as graph nodes |
| **Event record/wait nodes** | Synchronize graphs with streams via events |
| **Child graph nodes** | Nested graphs for modular construction |
| **Improved stream capture** | Better handling of complex capture scenarios |

### Conditional Nodes (New in CUDA 12.x)

```cpp
// Conditional graph nodes enable control flow
cudaGraphConditionalHandle handle;
cudaGraphConditionalHandleCreate(&handle, graph, 0, cudaGraphCondAssign);

cudaGraphNode_t node;
cudaConditionalNodeParams params;
params.handle = handle;
params.type = cudaGraphCondAssign;
params.size = 1;  // number of child graphs

cudaGraphAddConditionalNode(&node, graph, NULL, 0, &params);
```

### Device-Side Graph Launch

```cpp
__global__ void deviceGraphLauncher(cudaGraphExec_t graph) {
    cudaLaunchGraph(graph);  // Launch graph from device code
}
```

### Relevance to This Project

- **Training loop**: The entire forward + backward + update step could be captured as a CUDA Graph, eliminating the 0.6–4.0ms Python overhead identified in PLAN.md.
- **The PLAN.md notes**: "Write a `train_step_graph()` that captures fwd+bwd+update in a CUDAGraph." — CUDA 12.x's enhanced graph APIs make this more practical.
- **Conditional nodes**: Could be used for the quant warmup ramp (`quant_update_freq` steps) — different graph paths for quantized vs non-quantized steps.
- **Device-side launch**: For dynamic parallelism patterns where parent kernels need to launch sub-kernels.

---

## 11. Programmatic Dependent Launch (PDL)

### What It Is

PDL allows kernels to **declare dependencies on preceding kernels** at the programming level, enabling the runtime to overlap kernel execution when resources allow.

### API

```cpp
// Programmatic Dependent Launch (CUDA 12.0+)
// Kernel B can start before Kernel A finishes if resources allow
cudaLaunchKernelEx(&config, kernelA, args...);
// Declare dependency: kernelB depends on kernelA's output
cudaLaunchKernelEx(&config, kernelB, args...);
```

### How It Works

1. Kernel A launches and begins executing
2. Kernel B is launched with a dependency on Kernel A
3. The CUDA runtime can begin executing Kernel B's independent blocks while Kernel A's dependent blocks are still running
4. This creates a **software pipeline** overlapped at the block level

### Use Cases

- **Producer-consumer kernels**: First kernel produces data, second kernel consumes it
- **Multi-stage pipelines**: Data flows through multiple kernels with partial overlap
- **Attention computation**: Score computation (kernel A) feeds into softmax + value aggregation (kernel B)

### Relevance to This Project

- **SubQSA pipeline**: The compression branch (kernel 1) produces compressed K/V → selection branch (kernel 2) consumes them. PDL could overlap these kernels.
- **Forward + backward overlap**: In the training pipeline, parts of the backward pass could begin while the forward pass is still completing on later layers.

---

## 12. C++20 Device Features

### Supported Features

| Feature | Device Code | Host Code | Notes |
|---------|-------------|-----------|-------|
| **Three-way comparison (`<=>`)** | ✅ | ✅ | Full support with `auto operator<=>` |
| **`consteval`** | ✅ | ✅ | Immediate functions |
| **Designated initializers** | ✅ | ✅ | `{.x = 1, .y = 2}` |
| **`constexpr` virtual functions** | ✅ | ✅ | In constant evaluation context |
| **`constexpr` dynamic_cast** | ✅ | ✅ | In constant evaluation context |
| **Lambda templates** | ✅ | ✅ | `[]<typename T>(T t) {}` |
| **`std::span`** | ✅ | ✅ | Via libcu++ |
| **Concepts** | ✅ | ✅ | C++20 concepts |
| **Ranges** | ❌ | ✅ | Host only |
| **Modules** | ❌ | ✅ | Host only |
| **Coroutines** | ❌ | ✅ | Host only |
| **`std::format`** | ❌ | ✅ | Host only |

### Example: C++20 Three-Way Comparison in Device Code

```cpp
struct TensorView {
    float* data;
    int dims[4];
    
    auto operator<=>(const TensorView&) const = default;
    
    __device__ float& at(int b, int h, int t, int d) {
        return data[((b * dims[1] + h) * dims[2] + t) * dims[3] + d];
    }
};
```

### Example: Designated Initializers

```cpp
struct KernelConfig {
    int block_size;
    int grid_size;
    int shared_mem;
    bool use_tensor_cores;
};

__global__ void configured_kernel(KernelConfig cfg) {
    // ...
}

// Launch with designated initializers
KernelConfig cfg = {
    .block_size = 256,
    .grid_size = 1024,
    .shared_mem = 48 * 1024,
    .use_tensor_cores = true
};
```

### Relevance to This Project

- **Three-way comparison**: Useful for comparing tensor shapes, indices, and configuration values in device code.
- **Designated initializers**: Cleaner initialization of kernel configs and tensor descriptors.
- **Constexpr**: More compile-time computation for tile sizes, buffer offsets, and quantization parameters.
- **`--std=c++20`** must be passed to nvcc to enable these features.

---

## 13. Dynamic Parallelism v2

### What Changed

| Aspect | v1 (CUDA 11.x) | v2 (CUDA 12.x) |
|--------|----------------|----------------|
| **Implicit sync** | Parent waited for child automatically | **Explicit `cudaDeviceSynchronize()` required** |
| **API** | `cudaLaunchDevice()` | Direct kernel launch syntax `<<<>>>` |
| **Performance** | Higher overhead | Redesigned for lower overhead |
| **Graph integration** | Not supported | Device-side graph launch (`cudaLaunchGraph()`) |

### Example

```cpp
__global__ void child(float* data, int N) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x) {
        data[i] = sqrtf(data[i]);
    }
}

__global__ void parent(float* data, int N) {
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    child<<<gridSize, blockSize>>>(data, N);
    cudaDeviceSynchronize();  // REQUIRED in v2 (was implicit in v1)
}
```

### Key Changes

1. **No implicit synchronization**: Parent must explicitly call `cudaDeviceSynchronize()` — this is a breaking change from v1.
2. **Lower overhead**: The redesigned API has lower launch latency.
3. **Device-side graph launch**: Parent kernels can launch CUDA Graphs from device code.
4. **Better device-side memory management**: `cudaMallocAsync`/`cudaFreeAsync` available from device code.

---

## 14. Lazy Module Loading

### What It Is

Starting with CUDA 12.0, **lazy module loading is the default**. Instead of loading and JIT-compiling all PTX modules at application startup, modules are loaded on-demand when first needed.

### How It Works

| Mode | Behavior | Startup Time | First Kernel Time |
|------|----------|--------------|-------------------|
| **Lazy (default)** | Modules loaded on first use | Fast | May be slower (JIT compile) |
| **Eager** | All modules loaded at startup | Slower | Consistent |

### Configuration

```bash
# Disable lazy loading (force eager loading)
export CUDA_MODULE_LOADING=EAGER

# Check current mode
echo $CUDA_MODULE_LOADING
```

### Benefits

- **Faster application startup**: Only loads modules when needed.
- **Reduced memory usage**: Unused modules are never loaded.
- **Better for complex applications**: Applications with many kernel variants only load what they actually use.

### Trade-offs

- **First-launch latency**: The first time a kernel is called, it may incur JIT compilation overhead.
- **Profiling**: For accurate profiling, use eager loading to avoid JIT noise.
- **Debugging**: Eager loading is recommended during debugging.

### Relevance to This Project

- **Multi-architecture fatbinaries**: If building for both sm_75 and sm_90, lazy loading avoids loading the unused architecture's modules.
- **Kernel variants**: The project has multiple kernel variants (scalar, TC, packed). Lazy loading means only the actually-dispatched variant gets loaded.
- **PyTorch integration**: `torch.utils.cpp_extension.load_inline()` already uses JIT compilation. Lazy loading complements this by deferring module loading.

---

## 15. nvJitLink: JIT Link-Time Optimization

### What It Is

nvJitLink is a **standalone library for runtime linking of GPU device code**. It replaces the deprecated `cuLink*` APIs and adds support for Link-Time Optimization (LTO).

### Capabilities

| Capability | Description |
|-----------|-------------|
| **Input formats** | Host objects, fatbins, relocatable PTX, device cubins, PTX, LTO-IR, index files |
| **Output** | Linked cubin (loadable by `cuModuleLoadData`) |
| **LTO** | Link-time optimization across translation units when given LTO-IR |
| **JIT compilation** | If input lacks GPU assembly, it's compiled then linked |

### API

```cpp
#include <nvJitLink.h>

nvJitLinkHandle handle;
const char* options[] = {"-arch=sm_90", "-lto"};
nvJitLinkCreate(&handle, 2, options);

// Add input (LTO-IR, PTX, cubin, etc.)
nvJitLinkAddData(handle, NVJITLINK_INPUT_LTOIR, ltoir_data, ltoir_size, "module_name");

// Link
nvJitLinkComplete(handle);

// Get linked cubin
size_t cubin_size;
nvJitLinkGetLinkedCubinSize(handle, &cubin_size);
char* cubin = new char[cubin_size];
nvJitLinkGetLinkedCubin(handle, cubin);

// Load with CUDA Driver API
cuModuleLoadData(&module, cubin);
```

### Advantages Over cuLink*

| Aspect | cuLink* (deprecated) | nvJitLink |
|--------|---------------------|-----------|
| **LTO-IR support** | Deprecated | Full support |
| **Standalone library** | Part of CUDA Driver | Separate library |
| **Static linking** | Not possible | Static and shared available |
| **Cross-TU optimization** | Limited | Full LTO |

### Build with LTO

```bash
# Compile to LTO-IR
nvcc -arch=lto_90 -rdc=true -fatbin offline.cu

# Link at runtime with nvJitLink
# (see API example above)
```

### Relevance to This Project

- **Kernel compilation**: The project uses `torch.utils.cpp_extension.load_inline()` which JIT-compiles CUDA. nvJitLink could replace this for more efficient compilation.
- **Cross-kernel LTO**: With multiple kernel files (ternary matmul, elementwise, future SubQSA kernels), LTO could optimize across compilation units.
- **Smaller binaries**: LTO can eliminate dead code and inline across modules.

---

## 16. CUDA Minimal API (Driver API)

### What It Is

The CUDA Minimal API is a **reduced driver API** for applications that only need basic CUDA functionality. It provides a smaller runtime footprint.

### Core Minimal API Functions

| Function | Purpose |
|----------|---------|
| `cuInit()` | Initialize CUDA |
| `cuDeviceGet()` | Get device handle |
| `cuCtxCreate()` | Create context |
| `cuModuleLoadData()` | Load compiled module |
| `cuModuleGetFunction()` | Get kernel function |
| `cuLaunchKernel()` | Launch kernel |
| `cuMemAlloc()` / `cuMemFree()` | Device memory management |
| `cuMemcpyHtoD()` / `cuMemcpyDtoH()` | Memory transfers |
| `cuCtxDestroy()` | Cleanup |

### When to Use

- **Embedded systems**: Minimal footprint required
- **Library development**: Avoid heavy runtime dependencies
- **Custom frameworks**: Full control over CUDA lifecycle

### Relevance to This Project

For the Ultimate AI Model project, the **full CUDA Runtime API** (via PyTorch) is more appropriate than the minimal API. However, understanding the minimal API is useful for:
- Custom kernel loading and compilation utilities
- Standalone benchmarking tools that don't depend on PyTorch
- Understanding what PyTorch's CUDA integration does under the hood

---

## 17. Memory Synchronization Domains

### What They Are

Synchronization domains provide **finer-grained memory ordering** than full system fences. On Hopper (sm_90), they allow:
- Ordering memory accesses within a specific domain
- Avoiding unnecessary global fences when only local ordering is needed

### Fence Hierarchy

| Fence | Scope | Cost | When to Use |
|-------|-------|------|-------------|
| `__threadfence_block()` | Block | Lowest | Within a single block |
| `__threadfence()` | Device | Medium | Cross-block on same device |
| `__threadfence_system()` | System | Highest | Cross-device, host+device |
| **Domain-specific** | Configurable | Variable | Producer-consumer patterns |

### Hopper-Specific Domain Fences

On sm_90, the hardware supports memory synchronization domains that allow more efficient producer-consumer patterns:

```cpp
// Producer writes data, then signals via domain fence
data[idx] = computed_value;
__threadfence_domain();  // Domain-scoped fence

// Consumer waits for data within same domain
__threadfence_domain();
float val = data[idx];
```

### Relevance to This Project

- **TMA + barrier pipeline**: Domain fences complement TMA and async transaction barriers for precise ordering.
- **Multi-block attention**: When multiple blocks compute different attention branches, domain fences provide ordering without the cost of system-wide fences.
- **Fused kernel coordination**: Within a cluster, domain fences enable efficient block-to-block ordering.

---

## 18. Reduced Register Usage & Occupancy

### Hopper SM Occupancy Characteristics

| Parameter | H100 | A100 |
|-----------|------|------|
| **Max warps / SM** | 64 | 64 |
| **Register file / SM** | 64K 32-bit regs (256 KB) | 64K 32-bit regs (256 KB) |
| **Max registers / thread** | 255 | 255 |
| **Max blocks / SM** | 32 | 32 |
| **Shared memory / SM** | Up to 228 KB | Up to 164 KB |
| **FP32 cores / SM** | 128 | 64 |

### Key Insight: Register-to-Core Ratio Changed

The **ratio of SM registers to FP32 cores** changed from 1024 (A100) to **512** (H100). This means:
- Each FP32 core in H100 has access to fewer registers
- Register pressure is a bigger concern on H100
- Kernels that used few registers on A100 may need optimization on H100

### `__launch_bounds__` for Register Control

```cpp
// Tell the compiler to limit register usage
__global__ void __launch_bounds__(256, 2)  // maxThreads, minBlocks
my_kernel(float* data) {
    // Compiler will use at most 64K/(256*2) = 128 regs per thread
    // This ensures at least 2 blocks can run concurrently on one SM
}
```

### Register Optimization Strategies

1. **`__launch_bounds__`**: Hint the compiler to limit registers per thread.
2. **Shared memory tiling**: Move data from registers to shared memory when possible.
3. **Loop unrolling control**: `#pragma unroll 1` to prevent register explosion.
4. **Fewer intermediate variables**: Reuse registers explicitly.
5. **Occupancy calculator**: Use `cudaOccupancyMaxActiveBlocksPerMultiprocessor` to find optimal launch configs.

### Relevance to This Project

- **Ternary matmul kernels**: The existing kernels use WMMA with half2 unpacking. On Hopper, WGMMA may use different register patterns.
- **Fused kernels**: The planned `subqsa_combine_kernel` fuses multiple operations — register pressure could be a concern. Use `__launch_bounds__` to control.
- **Attention kernels**: The 3-branch SubQSA attention with gating needs many intermediate values. Careful register management is essential.

---

## 19. nvcc Flags for CUDA 12.x

### Essential Flags

| Flag | Purpose | Example |
|------|---------|---------|
| `-arch=sm_75` | Target Turing (T4) | `nvcc -arch=sm_75` |
| `-arch=sm_80` | Target Ampere (A100) | `nvcc -arch=sm_80` |
| `-arch=sm_89` | Target Ada Lovelace (RTX 4090) | `nvcc -arch=sm_89` |
| `-arch=sm_90` | Target Hopper (H100) | `nvcc -arch=sm_90` |
| `-arch=lto_90` | LTO-IR for Hopper | `nvcc -arch=lto_90` |
| `--std=c++20` | Enable C++20 in device code | `nvcc --std=c++20` |
| `-O3` | Maximum optimization | Standard |
| `-lineinfo` | Line info for profiling | Nsight compatible |
| `-G` | Debug info (disables optimizations) | Debugging only |
| `--use_fast_math` | Fast math intrinsics | Precision trade-off |
| `--expt-extended-lambda` | Extended device lambdas | `nvcc --expt-extended-lambda` |
| `--expt-relaxed-constexpr` | Relaxed constexpr | More constexpr in device code |
| `-rdc=true` | Relocatable device code | Required for dynamic parallelism |
| `--dlto` | Device Link-Time Optimization | Cross-TU optimization |
| `-t0` | Fast compilation | Debug builds |
| `--ptxas-options=-v` | Verbose PTX assembly | See register usage, SMEM |

### Multi-Architecture Fatbinary

```bash
# Build for multiple architectures
nvcc -O3 \
  -gencode arch=compute_75,code="sm_75" \
  -gencode arch=compute_80,code="sm_80" \
  -gencode arch=compute_90,code="sm_90,compute_90" \
  -o my_app my_kernel.cu -lcudart
```

### Shared Library for PyTorch (load_inline)

```bash
# Shared library for PyTorch JIT compilation
nvcc -O3 -arch=sm_90 --shared -Xcompiler -fPIC -o libkernel.so kernel.cu

# With C++20
nvcc -O3 --std=c++20 -arch=sm_90 --shared -Xcompiler -fPIC -o libkernel.so kernel.cu

# With LTO
nvcc -O3 --dlto -arch=lto_90 --shared -Xcompiler -fPIC -o libkernel.so kernel.cu
```

### Profiling Flags

```bash
# For Nsight Compute profiling
nvcc -O3 -arch=sm_90 -lineinfo -o test kernel.cu -lcudart

# See register/SMEM usage per kernel
nvcc -O3 -arch=sm_90 --ptxas-options=-v -o test kernel.cu -lcudart
```

---

## 20. Compilation & Multi-Architecture Support

### Architecture Support Matrix

| sm_ | Architecture | GPUs | CUDA 12.x Status |
|-----|-------------|------|------------------|
| sm_35 | Kepler | K20, K40 | ❌ **Dropped** in 12.0 |
| sm_50 | Maxwell | GTX 900 | ✅ Supported |
| sm_60 | Pascal | P100 | ✅ Supported |
| sm_70 | Volta | V100 | ✅ Supported |
| sm_75 | Turing | T4, RTX 2080 | ✅ Supported |
| sm_80 | Ampere | A100, RTX 3090 | ✅ Supported |
| sm_86 | Ampere | RTX 3080, A40 | ✅ Supported |
| sm_89 | Ada Lovelace | RTX 4090, L4 | ✅ **New in 12.0** |
| sm_90 | Hopper | H100 | ✅ **New in 12.0** |

### Driver Requirements

| CUDA Toolkit | Minimum Driver |
|-------------|----------------|
| CUDA 12.0 | ≥ 525.60.13 |
| CUDA 12.1 | ≥ 530.30.02 |
| CUDA 12.2 | ≥ 535.54.03 |
| CUDA 12.4 | ≥ 550.54.15 |
| CUDA 12.9 | ≥ 575.51.03 |

### Binary Compatibility

- Within the same CUDA major version (12.x), binaries are forward-compatible with newer minor versions.
- PTX forward compatibility: PTX compiled for `compute_90` can run on future architectures via JIT.

### LTO Workflow

```bash
# Step 1: Compile to LTO-IR
nvcc -arch=lto_90 -rdc=true -c module1.cu -o module1.ltoir
nvcc -arch=lto_90 -rdc=true -c module2.cu -o module2.ltoir

# Step 2: Link with device LTO
nvcc -arch=sm_90 --dlto module1.ltoir module2.ltoir -o app -lcudart
```

---

## 21. Relevance to Ultimate AI Model

### Current State (from PLAN.md)

- Kernels compile on CUDA 13.3, targeting **sm_75 (T4)**
- 11 kernel files in `kernels/packed_ternary/`
- Performance gap: 12× vs memory-bound minimum
- Python overhead: 0.6–4.0ms (major bottleneck)
- Next priorities: CUDA Graphs, kernel fusion, double-buffered SMEM

### Recommended Hopper Migration Path

#### Phase 1: Multi-Architecture Build (Low Risk)

```bash
# Add sm_90 to existing sm_75 build
nvcc -O3 \
  -gencode arch=compute_75,code="sm_75" \
  -gencode arch=compute_90,code="sm_90,compute_90" \
  --std=c++20 \
  --ptxas-options=-v \
  -o test_ternary kernels/packed_ternary/ternary_matmul.cu -lcudart
```

This ensures backward compatibility while enabling Hopper features.

#### Phase 2: TMA-Driven Async Data Movement (High Impact)

Replace manual data loading with TMA for:
- **Activation tile loading**: 2D tensor copies from global → shared memory
- **Weight tile loading**: Packed ternary weights loaded asynchronously
- **SMEM pipeline**: Double-buffered with TMA + async transaction barriers

Expected impact: **~2× kernel speedup** by freeing threads from data movement.

#### Phase 3: Thread Block Clusters + DSMEM (High Impact)

For the SubQSA attention pattern:
- Run compression, selection, and sliding window branches as separate cluster blocks
- Exchange results via DSMEM instead of global memory
- Use hardware-accelerated cluster barriers for synchronization

Expected impact: **~7× faster** inter-block data exchange for attention branches.

#### Phase 4: FP8 Tensor Cores (High Impact)

- Replace INT8 activation quantization with FP8 E4M3
- Use FP8 tensor cores for forward pass matrix multiplies
- Use Transformer Engine for the compression MLP and gate MLP

Expected impact: **2× throughput** over INT8 for activation processing.

#### Phase 5: CUDA Graphs for Training Loop (High Impact)

Capture the entire forward + backward + update as a CUDA Graph:
- Eliminates 0.6–4.0ms Python overhead
- Enables conditional nodes for quant warmup
- Device-side graph launch for dynamic parallelism patterns

Expected impact: **~0.1ms** Python overhead (from 0.6–4.0ms).

#### Phase 6: Programmatic Dependent Launch (Medium Impact)

Overlap compression and selection kernels with PDL:
- Compression kernel produces compressed K/V
- Selection kernel starts consuming while compression finishes
- Creates a software pipeline at the block level

Expected impact: **~1.5×** throughput improvement for the attention pipeline.

### Priority Matrix

| Feature | Impact | Effort | Priority |
|---------|--------|--------|----------|
| Multi-arch build | Low | Low | Quick win |
| CUDA Graphs | High | Medium | **P1** |
| TMA async copies | High | Medium | **P1** |
| Thread block clusters | High | High | **P2** |
| FP8 tensor cores | High | Medium | **P2** |
| DSMEM | High | High | **P2** |
| PDL | Medium | Medium | **P3** |
| WGMMA | Medium | Low | **P3** |
| Transformer Engine | Medium | Medium | **P3** |
| nvJitLink LTO | Low | Low | **P4** |
| Lazy loading | Low | None (default) | Auto |
| C++20 device features | Low | Low | **P4** |
| Dynamic Parallelism v2 | Low | Low | **P4** |

---

## 22. References

### Official NVIDIA Documentation

- [CUDA C++ Programming Guide (12.0)](https://docs.nvidia.com/cuda/archive/12.0.0/cuda-c-programming-guide/)
- [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/)
- [CUDA Toolkit Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/)
- [nvJitLink Documentation](https://docs.nvidia.com/cuda/nvjitlink/)
- [CUDA Toolkit 12.x Downloads](https://developer.nvidia.com/cuda-toolkit-archive)

### NVIDIA Blog Posts

- [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) — Comprehensive H100 architecture walkthrough
- [Transformer Engine Overview](https://docs.nvidia.com/deeplearning/transformer-engine/) — FP8 training and inference

### Vault Documentation

- [[cuda-12.x-reference]] — Existing CUDA 12.x reference in vault
- [[cuda-12.x-programming-guide]] — Existing programming guide
- [[cuda-12.x-compilation]] — Existing compilation guide
- [[cuda-13.x-reference]] — CUDA 13.x reference (successor)
- [[block_sparse_ternary]] — Block-sparse ternary matmul design
- [[cuda-kernel-fusion-design]] — Kernel fusion design for this project

### Related Research

- [BitNet b1.58 2B4T](https://arxiv.org/abs/2504.12285) — Ternary quantization
- [NSA: Native Sparse Attention](https://arxiv.org/abs/2502.11089) — Sparse attention design

---

*This document was compiled from NVIDIA official documentation, developer blog posts, the Hopper tuning guide, and the vault's existing CUDA research notes.*
