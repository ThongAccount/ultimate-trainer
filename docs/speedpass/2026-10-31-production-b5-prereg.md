# PRE-REGISTRATION 2026-10-31 — production-config flip budget B=5

**Written and committed BEFORE the T4 allocation.** Nothing below may be
changed after seeing the run's numbers. The synthetic gate is closed
(only B=5 passes V+M+C, sweep @ 80fdbec); this run answers the two open
questions on the REAL config: does B=5 still damp the limit cycle on real
FineWeb text, and does the shipped speedup survive the flip counting?

## Protocol (fixed)

- One Colab T4 allocation, `scripts/colab_official_suite.py` @ the commit of
  this prereg, `--nsteps 200 --flip-steps 1000 --data real --flip-budget 5`.
- Arms: custom_custom / cublas_custom / cublas_cublas (as 2026-10-26), seed 0,
  FineWeb `data/tokenized/samples_512.pt`, 3 fixed 32x512 batches cycled by
  `step % 3`, median of steps 5..199 for timing.
- Production layers get `flip_budget=5` via `PT_FLIP_BUDGET` (train_gigatoken
  `FLIP_BUDGET`); the flip gate phase runs the synthetic 512x512 layer with
  `--flip-budget 5` and the pre-registered exempt-control semantics
  (`arm_flip_budget`, d7aa1d1). nsteps 200 (not 300) to fit the Colab window;
  the A/B comparability points (loss at steps 1/50/100/final, median ms) are
  unchanged in kind.
- Smoke pass runs first (same VM, tiny params); a smoke failure aborts before
  the official measurements.

## Acceptance criteria (all pre-registered, evaluated from artefacts only)

- **P0 (sanity, synthetic gate)**: PHASE 2 flip gate at B=5 passes V+M+C
  (reproduces B05x @ 55217f6: V 0.00494 >= 2.44e-4, M 0.0294 <= 0.10,
  C 1.035 >= 0.50).
- **P1 (damping on real config)**: in EVERY A/B arm, the cumulative production
  flip count decays late vs mid run: `flips_final - flips@99 < flips@99 - flips@49`
  (budget B=5 exhausts per-weight budgets inside 200 steps; the unbudgeted
  2026-10-26 run sustained instead), AND no late loss blowup:
  `loss_final <= loss@100 + 1.0`.
- **P2 (speed survives)**: `AB_VERDICT` ship bar unchanged —
  `reduction_pct >= 5.0` (custom_custom -> cublas_cublas), AND shipped-arm
  absolute sanity: cublas_cublas median_step_ms <= 1.25 x 510.9 ms
  (2026-10-26 same-VM number; cross-VM drift caveat noted there).
- Suite rc=0 requires P0 + AB pass; P1 is reported per arm either way.

## Interpretation rule

If P0+P2 pass and P1 holds in all arms: B=5 is a viable production damping
setting — the limit cycle is broken on real text without losing the speedup.
If P1 fails (flips sustain) while P0 passes: the synthetic gate does not
transfer to the production config. If P2 fails: flip counting costs too much
at production shape. Any criteria change after the run is invalid, same as
the 2026-10-26/29 pre-registrations.
