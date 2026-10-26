# Official suite — whole-step A/B + flip-rate trajectory gate (2026-10-26)

Session 32. This closes the two Phase-2 acceptance items that were blocked on
a T4 from 2026-10-21 to 2026-10-26 (Colab allocation had answered
`Service Unavailable`; the account was at 0.00 compute units).

**Headline:** the whole-step A/B is **PASSED** — the shipped configuration is
**9.51× faster than the same-VM baseline** (4,860.5 ms → 510.9 ms, 89.49%
end-to-end step-time reduction, ship bar ≥5%). The flip-rate trajectory gate
is **FAILED**, and the failure is a real finding, not a tooling artefact:
**the ternary optimizer does not settle — it enters a limit cycle.** Removing
the gradient noise does not reduce the flip activity; it plateaus.

Both results are from ONE T4 allocation, one repo clone, one set of JIT
builds (`scripts/colab_official_suite.py`).

---

## 1. Conditions (shared by every number in this document)

| What | Value |
|---|---|
| GPU | Tesla T4 (hosted Colab VM), capability (7, 5) — asserted in the driver |
| Commit measured | `015763aff0` (all three arms + both gate backends) |
| Repo state | `origin/master` at that commit, cloned once, `reset --hard` |
| Model | production 6-layer ternary transformer (`train_gigatoken.build_model`): K=1024, 6 layers, VOCAB=50272, threshold=32 |
| Batch / sequence | B=32, SEQ=512 → 16,384 tokens per batch; tok/s counts 32×511 = 16,352 (the driver's convention, unchanged) |
| Data | **real** — committed FineWeb tokenized samples (`data/tokenized/samples_512.pt`, custom BPE, vocab 32,768; max token id 32,766 < VOCAB 50,272). 3 fixed batches of 32×512, cycled deterministically (`step % 3`), identical across all arms |
| Steps | 300 per arm; warm-up 5 discarded; statistic = median of the remaining 295 wall-clock step times (each timed with `time.perf_counter()` around the step plus a `torch.cuda.synchronize()`) |
| Seed | `torch.manual_seed(0)` before model build and before data construction — identical across arms |
| Ship bar | ≥5% end-to-end step-time reduction (the Researcher's bar, protocol item 3) |

`DATA_MODE=real` is the primary; `DATA_MODE=synthetic` (the earlier protocol's
single fixed random-token batch) is still available in the script.

---

## 2. Whole-step A/B — three arms, one VM, 300 steps each

`scripts/colab_step_ab.py`, run as three separate subprocesses (backend is
read at import time, so the arms are isolated in separate CUDA contexts).

| arm | `PT_BACKEND` | `PT_UPDATE_BACKEND` | median step ms | tok/s | vs baseline |
|---|---|---|---|---|---|
| `custom_custom` | custom | custom | **4,860.5** | 3,364 | — (baseline) |
| `cublas_custom` | cublas | custom | **1,696.8** | 9,637 | 2.86× |
| `cublas_cublas` | cublas | cublas | **510.9** | 32,005 | **9.51×** |

**Verdict:** `reduction_pct = 89.49`, `speedup_same_vm = 9.51x`,
`speedup_vs_historical_6619 = 12.96x`, `passed = True` (ship bar 5%).

The three arms reproduce the campaign's whole-step trajectory
(`HANDOFF.md` §31) end to end in a single run:

```
6,619 ms  (Modal, historical reference — different VM/torch/python/power)
  -> 4,860.5 ms  custom_custom        (this run, same VM)
  -> 1,696.8 ms  cublas_custom        (fwd/dX re-host + forward-routing fix)
  ->   510.9 ms  cublas_cublas        (shipped: + update reduction re-host)
```

**Cross-check against session 31:** 1,696 ms → **1,696.8 ms** (the forward
fix reproduces) and 519 ms → **510.9 ms** (the update re-host reproduces).
The baseline arm is 4,860.5 ms here against 5,443.6 ms measured on Colab in
session 29 and 6,619 ms on Modal — different VMs, so only the **same-VM**
ratio above is acceptance-grade. The 12.96× against the 6,619 ms figure is
reported for continuity and carries that caveat.

### 2.1 Loss trajectory (real text — new)

| step | custom_custom | cublas_custom | cublas_cublas |
|---|---|---|---|
| 1 | 11.0625 | 11.0625 | 11.0625 |
| 50 | 33.6875 | 33.6875 | 33.6875 |
| 100 | 40.4375 | 40.4375 | **51.3125** |
| 300 (final) | 22.0312 | 22.0312 | **35.5312** |

`nan = 0` in all three arms.

Three things follow, and all three are new:

1. **The step-99 NaN was a synthetic-data artefact.** With real FineWeb
   tokens the 300-step run never NaNs. The earlier "loss track blocked on
   data" item is therefore *not* blocked any more — the loss track is
   measurable, and it is bad.
2. **The fwd/dX change is bit-identical in its loss trajectory.**
   `custom_custom` and `cublas_custom` agree to the last digit at every
   printed point, confirming session 31's "trajectory bit-identical" claim
   for the forward fix on real data.
3. **The shipped update re-host diverges from the custom path from step 100
   on** (51.3125 vs 40.4375; 35.5312 vs 22.0312 final). This is the
   *disclosed* counter jitter (session 31: "no longer bit-identical"), now
   measured on real text instead of synthetic tokens. It is a documented
   consequence of the cuBLAS dW summation order, not a new defect — but it
   is a real difference in trajectory, so the two configurations are NOT
   interchangeable for training.

**None of the three arms converges.** The loss rises from 11.06 to ~40–51 by
step 100 and then partially recovers by step 300. Section 3 explains why.

---

## 3. Flip-rate trajectory gate — FAILED (limit cycle detected)

`scripts/colab_flip_rate_gate.py`. The criteria were **pre-registered before
any measurement** in `tools/flip_rate_verdict.py` (committed in `f87ad09`,
measured in `015763a`), with 6 CPU tests in `tests/test_flip_rate_verdict.py`
pinning them.

### 3.1 Why the gate exists

The optimizer has no learning rate and no damping. The counter accumulates
`sign(dW)`; when `|counter| > threshold` the weight flips and the counter
resets to 0. A weight that keeps receiving a *consistent* gradient therefore
flips every `threshold` steps forever — a **limit cycle**. Its observable
signature is a flip rate that does **not** decay.

Expected magnitudes, from the counter mechanics alone (not from measurements):

| regime | flip rate |
|---|---|
| limit cycle (consistent residual gradient) | ~1/T = 3.125% |
| pure zero-mean gradient noise (random walk) | ~1/T² = 0.098% |
| settled (dW = 0, counter frozen) | 0 |

### 3.2 Setup

Single `PackedTernaryLinear(512 → 512)`, threshold **32** (the production
threshold), B=512, **262,144 weights**, 1,000 steps, seed 1234. The target is
the layer's *own* output before a perturbation, so it is exactly representable
on the ternary grid and a healthy optimizer can settle on it. 5% of the
weights are moved one grid step, so a real residual exists to correct. Per
step the update op is called directly with `dY = dY_true + noise`, so the
noise enters at the **gradient** level — the same `dY` the production
backward hands the update kernel.

Noise schedule: **main** arm 1.0 → 0.3 → 0.0 over the warm-up / noisy /
clean windows; **control** arm 1.0 throughout. The control calibrates what
"flip activity that does not go away" looks like, so the main arm's decay
cannot pass vacuously. Windows: warm-up steps 0–299, noisy 300–599, clean
600–999.

Flips are counted **exactly**: the packed weight words are compared field by
field (2 bits per ternary value, 16 values per int32 word) — not by
inspecting counters, which reset on a flip.

### 3.3 Result

| arm | update backend | noisy window | clean window | ratio clean/noisy |
|---|---|---|---|---|
| main | custom | 0.014324315 | 0.015498409 | **1.08197** |
| main | cublas (shipped) | 0.014324315 | 0.015498409 | **1.08197** |
| control | custom | 0.012001432 | 0.012424068 | 1.03522 |
| control | cublas (shipped) | 0.012001432 | 0.012424068 | 1.03522 |

| criterion | requirement | value | verdict |
|---|---|---|---|
| V — non-vacuity | main noisy ≥ 0.25/T² = 0.000244141 | 0.014324315 | **PASS** |
| M — decay when noise removed | main clean/noisy ≤ 0.10 | 1.08197 | **FAIL** |
| C — control sustains flips | control late/early ≥ 0.50 | 1.03522 | **PASS** |

**`FLIPRATE_GATE: FAIL` for both update backends.**

Per-100-step trajectory (main arm, identical for both backends):

| step | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900 | 1000 |
|---|---|---|---|---|---|---|---|---|---|---|
| flip rate | 0.00490 | 0.00980 | 0.01115 | 0.01369 | 0.01449 | 0.01480 | 0.01523 | 0.01547 | 0.01561 | 0.01569 |
| residual RMS | 2.06e-6 | 2.74e-6 | 2.88e-6 | 3.17e-6 | 3.34e-6 | 3.41e-6 | 3.49e-6 | 3.55e-6 | 3.59e-6 | 3.61e-6 |

### 3.4 What this means

**The flip rate rises monotonically and saturates at ≈1.57%, and removing the
gradient noise does not reduce it** — it goes *up* slightly (ratio 1.082,
criterion M requires ≤0.10). The residual grows over the same interval.

This is the limit-cycle signature: the weights keep flipping because the
gradient keeps pointing the same way at the ternary grid optimum, not because
of noise. The control arm confirms the noise is not what sustains it — with
constant noise the rate also plateaus (1.24%) instead of decaying, so the
"noise-driven floor" is a plateau, not a decay.

Two further facts:

- **The custom and cuBLAS update backends are bit-identical in flip
  behaviour** — every bucket, every window, every rate. The shipped
  re-host does not change the optimizer's dynamics; it only changes how
  `dW` is computed, and at these magnitudes the sign pattern is the same.
  (This is consistent with session 31's "W bit-exact vs v2_64", counter
  diverging at 3/2 of 4.2M elements.)
- **This corroborates §2.1.** The production run's loss rising and partially
  recovering is the same phenomenon seen at production scale: the optimizer
  overshoots and oscillates rather than settling.

**Scope of the claim.** The gate ran a single 512×512 layer, not the 6-layer
production model. The counter/flip mechanism is per-weight, so the
*mechanism* transfers; the production model's dynamics may differ in detail.
What the gate establishes is that **the mechanism has no settling behaviour
at the production threshold** — there is no damping that would stop the
oscillation once the residual is small.

### 3.5 What this does NOT mean

- It is not an implementation bug in the re-host: both backends behave
  identically, and the failure is a property of the sign/counter rule.
- It is not a claim that the optimizer cannot learn — the layer did reduce
  its residual relative to the perturbed start, and §2.1 shows the loss does
  come back down by step 300. The claim is narrower: **it does not settle.**
- The gate's thresholds were fixed before the measurement and are not
  re-tuned against these numbers. The verdict is reported as it came out.

---

## 4. Tooling lessons (three T4 allocations)

Recorded because they cost real credits and will cost them again.

1. **`__file__` does not exist under `colab run`.** It executes the script
   inside a Jupyter kernel. A script that references `__file__` dies
   immediately with `NameError`.
2. **A child process that INHERITS stdout is invisible to the CLI.** Under a
   Jupyter kernel the CLI captures what the *kernel prints*; a child writing
   to the inherited fd 1 writes to the kernel server's descriptor instead.
   On 2026-10-26 the full official run **completed** (PHASE1_RC=0,
   PHASE2_RC=1) and every number was lost. Fixed by
   `tools/stream_runner.py`: run the child with an explicit PIPE and
   re-print each line from the kernel side (`readline()`, not iteration —
   iteration uses a read-ahead buffer that delays lines). Every phase now
   also writes its artefacts to `results/` on the VM, and the artefacts were
   downloaded as a second line of defence.
3. **The smoke pass earned its keep.** `scripts/colab_official_suite.py`
   runs the *same* scripts with tiny parameters on the *same* VM first (so
   the JIT builds are paid once). It caught a CPU/GPU device mismatch in the
   gate's perturbation mask after ~10 minutes of GPU, instead of after ~50
   minutes of official run. Keep the smoke.

Also fixed: `colab_pytest.py` — the correctness gate still cloned the
**deleted** `chore/speedpass` branch and hard-required a `GITHUB_TOKEN`, so
it could not run at all. It now tracks `master` and clones anonymously (the
repo is public).

---

## 5. Artefacts

Committed under `results/`:

| file | contents |
|---|---|
| `results/whole_step_ab.json` | the three arms, their medians, tok/s, loss points, the commit, the ship bar |
| `results/flip_rate_gate.log` | the gate's full output, including every per-100-step bucket |
| `results/flip_rate_gate.json` | the same as structured data, with the exit code |
| `results/official_suite.log` | the complete transcript of the run (smoke + both phases) |
| `results/official_suite.json` | the transcript plus the phase return codes and the verdict |

Raw VM-side copies were downloaded with `colab download` before the session
was released.

## 6. What is next

- The limit cycle is now a **measured, open problem**, not a hypothesis. The
  next candidate is damping the counter (a dead zone, a learning-rate-like
  counter scale, or a flip budget) — any change must be judged on the
  trajectory, not just on the flip rate, and must keep the shipped
  speedup.
- The production configuration does not converge (§2.1). Whatever fixes the
  limit cycle has to be judged against the loss trajectory on real text,
  which is now measurable — the "loss track blocked on data" item is closed.
- Phase 3 (custom-kernel work) remains blocked by design: ncu is
  permission-gated on hosted VMs, and the update reduction is already
  re-hosted on cuBLAS.
