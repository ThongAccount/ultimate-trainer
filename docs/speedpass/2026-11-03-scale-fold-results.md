# Results: scale fold into W16 (2026-11-03) — FAIL, not shipped

Branch `feat/scale-fold` @ b681d2b7, prereg `2026-11-02-scale-fold-prereg.md`,
validation ut-scalefold (one T4, cublas_cublas arm). Artefact
`results/scale_fold.b681d2b7.json` (rc=1, valid negative).

- pytest **104 passed** — incl. the new `test_scale_fold_parity`: the folded
  forward IS bit-exact vs post-mul outside fp16 subnormals, and full-step
  loss at steps 1/29 was identical fold-on vs fold-off (11.0625 / 11.054688).
- But fold-on median **2365 ms vs 512 ms fold-off (+361%, σ_null 2.8%)** and
  the loss froze at its step-29 value (fold-off reached 9.29 by step 59) →
  flip/update trajectory diverged. Gate FAIL on both perf and bit-stability.

Root cause not isolated (session reaped before log download): per-call
`W16 * s` allocates a fresh tensor, defeating decode_into's persistent
scratch buffer (and any capture assumptions), and the update path under fold
no longer reproduces master's flip trajectory. The measured prize was only
~30 ms (5%) of the 608 ms shipped step; the naive fold is a large loss.

**Decision: branch not merged, item #5 closed as measured-negative.**
If revisited, the scale must be folded at decode time (scale inside the
decode kernel's output write), not as a post-decode mul.
