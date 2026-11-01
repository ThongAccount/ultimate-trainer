# RESULTS 2026-11-01 — production B=5 confirmation at nsteps=300

Run: one Colab T4, `scripts/colab_official_suite.py` @ **a7fa910c6f**,
`--nsteps 300 --flip-steps 1000 --data real --flip-budget 5`, judged against
the pre-registered criteria in `2026-10-31-production-b5-300step-prereg.md`
(committed before the allocation). Artefacts: `results/*.B05prod300.*`.
Suite: `OFFICIAL_SUITE: PASS ab_rc=0 flip_rc=0`. Session `ut-b5-300b`.

## P0 — synthetic flip gate at B=5: PASS, bit-reproduces the JIT runs

PHASE 2 (1000 steps, both backends): V 0.0049401, M 0.0294108, C 1.03524,
F0/F1/F2 pass (F1 0.00937492), `FLIPRATE_GATE: PASS` rc=0 — **bit-identical**
to B05x @ 55217f6 and B05prod @ 4957fdd (all built by on-VM JIT; this run's
loss@49/99 also match 11.06/10.42/9.20 exactly). Determinism across VM
rebuilds confirmed.

## P1 — damping on the real config at 300 steps: PASS in all three arms

All arms bit-identical (loss_1 11.0625, loss_50 9.2891, loss_100 10.2344,
loss_final 9.1250, flips_final 3015, nan=0):

- Flip-rate decay: sampled cumulative flips @49=1858, @99=2274, @299=3015.
  Per-step rate mid-window (50-99) = 416/50 = **8.32/step**; late window
  (100-299) = 741/200 = **3.71/step** — decays ~2.2x, consistent with the
  150-step run's late rate (198/50 = 3.96/step). Budget B=5 keeps exhausting.
  (Prereg named flips@149/@249; the driver samples 1/49/99/final at
  nsteps=300, so P1 is evaluated on the available sampled points 99/299 —
  substitution noted here, criteria otherwise unchanged.)
- Loss: `loss_final 9.1250 <= loss@100 10.2344 + 1.0` and 9.1250 is the run
  minimum (11.06 -> 10.42 -> 9.29 -> 10.23 -> 9.125; oscillation amplitude
  ~1.0 vs the unbudgeted 11 -> 40 -> 22 swing).

## P2 — speed survives: PASS

`AB_VERDICT`: reduction_pct **87.71** >= 5.0 ship bar; same-VM speedup
**8.13x** (4666.7 -> 573.7 ms); shipped arm 573.7 ms <= 1.25 x 510.9 ms
(638.6). 11.54x vs the historical 6619 ms Modal figure (cross-VM, reference
only). Flip counting at B=5 costs nothing measurable at production shape.

## Kernel-build note (prebuilt pipeline)

This run's kernels were VM-JIT-built: the first `kernels-latest` bundle
(8d1e89a) shipped without `.o`/`.ninja_deps`, and torch 2.11's cache hit is
ninja mtime-based, so the VM recompiled once (smoke paid it; provenance
`UT_PREBUILT=0` in child artefacts, parent printed `PREBUILT: OK` for the
download itself). Fixed in 268907b (bundle now ships .o/.ninja_deps; fetch
touches .so newer than sources) — the release `kernels-latest` @ 268907b is
16.7 MB with 23 .so. A dedicated fresh-VM smoke validating zero-nvcc loading
follows this commit.

## Conclusion

B=5 per-weight flip budget breaks the limit cycle on the real FineWeb config
at the full 300-step baseline length — late-window flip rate decays, final
loss is the run minimum, and the ~8x same-VM speedup survives. The 150-step
and 300-step production runs agree on every shared comparability point.
