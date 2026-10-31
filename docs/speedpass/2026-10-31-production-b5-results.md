# 2026-10-31 — production-config flip budget B=5: RESULTS (prereg: 2026-10-31-production-b5-prereg.md)

Run: one Colab T4, `colab_official_suite.py` @ `4957fdd4`, `--nsteps 150
--flip-steps 1000 --data real --flip-budget 5`, smoke + PHASE1 A/B (3 arms,
FineWeb samples_512.pt, seed 0) + PHASE2 flip gate. **OFFICIAL_SUITE: PASS**
(ab_rc=0 flip_rc=0). Artefacts: `results/{whole_step_ab,official_suite,flip_rate_gate}.B05prod.{json,log}`.

## Pre-registered criteria verdict

- **P0 synthetic gate at B=5 — PASS**: V 0.0049401 >= 2.44e-4, M 0.0294 <= 0.10,
  C 1.0352 >= 0.50 (exempt control), F0/F1/F2 all pass; `FLIPRATE_GATE: PASS`
  rc=0. Bit-reproduces the B05x sweep point @ 55217f6.
- **P1 damping on the real production config — PASS in every arm**:
  cumulative production flips step49=1858, step99=2274, step149=2472, so
  late-window flips 2472-2274=198 < mid-window 2274-1858=416 (budget B=5
  exhausts and the flip rate decays); loss_final 9.1562 <= loss@100 10.2344+1.0.
  All three arms bit-identical (same losses, same flips_final=2472) — damping
  is backend-independent. For contrast, the unbudgeted 2026-10-26 run
  oscillated (loss 11.06 -> 40.4 @100 -> 22.0 @300); here loss decreases
  monotonically 11.06 -> 10.42 -> 9.20 -> 9.16 over 150 steps.
- **P2 speed survives — PASS**: `AB_VERDICT` reduction_pct=88.33 >= 5.0 ship
  bar; same-VM speedup 8.57x (4496.6 -> 524.8 ms); shipped arm 524.8 ms <=
  1.25 x 510.9 ms (2026-10-26 reference). Flip counting at B=5 costs nothing
  measurable at production shape (custom_custom 4496.6 ms vs 4860.5 ms at
  nsteps=300 unbudgeted — same or better).

## Conclusion

B=5 per-weight flip budget breaks the ternary optimizer's limit cycle on the
REAL FineWeb production config (loss now decreases instead of oscillating;
late-window flips decay as budgets exhaust) while keeping the shipped speedup
(88% step-time reduction vs custom baseline). The synthetic gate result
transfers to production. Note nsteps=150 (window-limited; prereg amended
before any measurement); a 300-step confirmation is the natural follow-up.
