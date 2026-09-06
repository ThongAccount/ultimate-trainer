# Session 9 (2026-09-06) — dX dataflow, GPU residency experiment, fusion decision

## Phase 1: dX dataflow (verified in code)

```
                         per layer i  (backward order: head → fc2i → fc1i)
─────────────────────────────────────────────────────────────────────────
 dY_layer(l)  ──►  backward_dx_tc_64/32          W_packed  (read, decoded)
      │                │
      │                ▼
      │            dX  [B,K] half  ──►  autograd  ──►  becomes dY of layer l-1
      │                          (also: layer-0 dX is unused by any update —
      │                           embedding grad comes from the embedding's
      │                           own backward, not from fc1's dX)
      │
      ├──►  update_tc_v2 (TC32):  dY ⊗ XR^T → dW  → counter ±sign → bit flip
      │       inputs: dY (this layer), X (this layer's saved input), W_packed
      └──►  X is saved from forward (ctx.X_saved), resident in VRAM
─────────────────────────────────────────────────────────────────────────
```

### dX consumers — verified list

1. **Autograd chain**: `PackedTernaryLinearFn.backward` returns dX (packed_linear.py:169);
   it becomes the incoming gradient of the *previous* op (plain tensor, global VRAM).
2. **Next layer's backward**: that dX tensor is the dY input of the previous
   layer's `backward()` — read by *its* dX kernel and *its* update kernel.
3. That's it. dX is **not** consumed by the *own* layer's update kernel —
   the update kernel reads `dY` and `X`, never `dX`. dX of each layer is read
   exactly twice, both reads downstream (next layer's two kernels).

### Answer to the driving question

**No — dX is not single-consumer-by-update; update never reads dX at all.**
The tempting "fuse update into dX epilogue" (noted in session 8 handoff) is
*not* a traffic win for dX: dX is already written once to global and read
exactly twice by the next layer; an in-register handoff would only matter if
update itself consumed dX. It doesn't.

## Phase 2: GPU-residency experiment (Modal T4, model 6×1024, batch 16384)

Instrumented one full step with torch.profiler runtime-API counts:

| event | count/step |
|---|---|
| cudaMemcpy (any) | **0** |
| cudaMemset | 1 |
| cudaDeviceSynchronize | 2 |
| cudaLaunchKernel | 175 |

Peak memory: 7.11 GiB alloc / 8.73 GiB reserved (15 GiB card) — everything fits.

Three configurations, 10 steps each, same model:

| config | ms/step | tok/s |
|---|---|---|
| A baseline (sync + loss.item() every step) | 3685.7 | 4,445 |
| B fully resident (no sync, item at end) | 3685.0 | 4,446 |
| C no sync, item every step | 3687.7 | 4,443 |

**B/A = 1.000. Conclusion: zero memcpys, sync/readback cost ≈ 0.0–0.1%.
The step is ~100% GPU kernel time. Memory movement is NOT the bottleneck.**

## Phase 3: fusion decision

### Why not "update consumes dX" fusion

As above — false premise. Skipping.

### The real shared operand is dY

The dX kernel and the update kernel of the *same* layer both stream the full
dY tile and the same saved X. A single fused kernel could in theory read
each dY/X tile once instead of twice, halving those reads. Head layer:
dY = 16384×50272×2B = 1.6 GB per read.

**But**: sessions 7+8 showed the update kernel is sync/latency-bound inside
its 16-deep WMMA loop (halving traffic changed nothing), and the dX kernel
at TC64 is bound by its own loop structure, not DRAM. Removing ~1.6 GB of
head-dY traffic (≈5 ms of the 550 ms head update at 320 GB/s peak) plus
12 MLP dYs (167 MB × 12 = 2.0 GB, ≈6 ms) saves ≪1% of step time even at
perfect bandwidth attribution. The existing fused kernel
(`gemm_fused_backward_update.cu`) also showed atomic-contention limits, and
it computes dX via slow non-TC path.

**Decision: do not implement dX+update fusion.** Expected gain ≤1%,
risk high (new kernel, new correctness surface, register pressure from
holding two accumulators).

### Where the remaining time actually goes (session 7 profile)

| phase | ms/step | share |
|---|---|---|
| fwd GEMM TC64 (13 launches) | 1371 | 36% |
| update TC32 (13) | 1142 | 30% |
| bwd dX TC64 (13) | 1061 | 28% |
| everything else | ~200 | 6% |

All three are WMMA kernels with 16-wide K-loops. The common structural cost
is the per-16-row smem reload + sync chain. Sessions 3–8 squeezed the obvious
micro-optimization space (coalescing, vectorization, tile dedup, sub-tiling)
to exhaustion on this T4:

- coalesced loads: **won** (−44% kernel)
- vectorized stores: **won** (−7..18%)
- 64×64 dX dispatch: **won** (+9.8% e2e)
- tile dedup: **lost** (+18% latency) — reverted
- kSub=4 batch sub-tiles: **lost** (2.2×) — reverted
- 64×64 update: correct but **neutral**

### Remaining structurally-plausible ideas (recorded, not committed)

1. **Deeper WMMA tiling of dX**: dX kernel is 4 frags/warp over a single 16-K
   slice; a 32-K slice (two MMAs between syncs) halves syncs without the
   20 KB smem blowup of kSub=4 in update (dX needs only 2×6 KB tiles).
2. **Reduce the launch count**: 39 kernel launches of the three big kernels;
   a megakernel per layer could merge dX+update loops but keeps each loop's
   latency chain — no theoretical gain from fusion alone.
3. **Larger-batch regime**: at K=1024 tiles the launch overhead is already
   amortized; scaling batch (B=32768) would raise arithmetic intensity in the
   update kernel's buried loops — untested, training-hyperparameter territory.

### Session output

- `tests/experiment_gpu_residency.py` + `modal_residency.py` (committed harness)
- This doc. No kernel changes; no e2e change expected (trainer untouched at
  1baa097…7602bcc; production kernels unchanged).
