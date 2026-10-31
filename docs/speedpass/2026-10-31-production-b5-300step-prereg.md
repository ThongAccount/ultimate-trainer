# PRE-REGISTRATION 2026-10-31 — production B=5 confirmation at nsteps=300

**Written and committed BEFORE the T4 allocation.** This is the natural
follow-up named in `2026-10-31-production-b5-results.md`: the 150-step run
(@ 71310fa) closed the open question (B=5 breaks the limit cycle on the real
FineWeb config, 88.3% step-time reduction, no speed loss). This run extends
the same protocol to nsteps=300 — the step count of the original 2026-10-26
unbudgeted baseline — and, as a side effect, validates the new CI-prebuilt
kernel path end-to-end.

## Protocol (fixed)

- One Colab T4 allocation, `scripts/colab_official_suite.py` @ the commit of
  this prereg, `--nsteps 300 --flip-steps 1000 --data real --flip-budget 5`.
- Kernels come from the `kernels-latest` release built by
  `.github/workflows/build-kernels.yml` at this commit
  (`tools/fetch_prebuilt.py`; `PREBUILT: OK` must appear in the transcript,
  `UT_PREBUILT=1` in artefact provenance). If prebuilt fetch fails the run
  falls back to JIT and the transcript says `PREBUILT: SKIP` — either way the
  run is valid; the prebuilt-vs-JIT equivalence is judged by P0 below.
- Arms, data, seed, comparability points: unchanged from
  `2026-10-31-production-b5-prereg.md` (custom_custom / cublas_custom /
  cublas_cublas, seed 0, FineWeb samples_512.pt, median of steps 5..299).
- Smoke pass runs first (same VM, tiny params); smoke failure aborts before
  official measurements. Flip gate phase: synthetic 512x512 layer,
  `--flip-budget 5`, exempt-control semantics (`arm_flip_budget`, d7aa1d1).

## Acceptance criteria (unchanged in kind from the 150-step prereg)

- **P0 (sanity + prebuilt-equivalence)**: PHASE 2 flip gate at B=5 passes
  V+M+C and **bit-reproduces** the JIT-built values (B05x @ 55217f6 /
  B05prod @ 4957fdd): V 0.00494 >= 2.44e-4, M 0.0294 <= 0.10,
  C 1.035 >= 0.50, F0/F1/F2 pass, rc=0. Any deviation means the prebuilt
  bundle does not match the sources — report as a prebuilt-path failure,
  not a damping result.
- **P1 (damping on real config, 300 steps)**: in EVERY A/B arm,
  `flips_final - flips@249 < flips@249 - flips@149` (late-window decay;
  thresholds scaled from the 150-step prereg's 49/99/149 points) AND
  `loss_final <= loss@250 + 1.0`, AND final loss is the run minimum or
  within 0.5 of it (2026-10-31 wording fix: report oscillation amplitude,
  never claim strict monotonicity).
- **P2 (speed survives)**: `AB_VERDICT` `reduction_pct >= 5.0`
  (custom_custom -> cublas_cublas) AND cublas_cublas median_step_ms
  <= 1.25 x 510.9 ms (2026-10-26 same-VM reference, cross-VM caveat).
- Suite rc=0 requires P0 + AB pass; P1 is reported per arm either way.

## Interpretation rule

Same as the 150-step prereg: P0+P2 pass with P1 in all arms -> B=5 is a
viable production damping setting confirmed at the baseline step count.
Any criteria change after the run is invalid.
