# Results: dW GEMM algo autotune (2026-11-03) — NEGATIVE, not shipped

Branch `feat/dw-algo` @ d4814f3 (bundle `kernels-dw-algo`, CI green), prereg
`2026-11-03-dw-algo-prereg.md`, session ut-dwalgo. Artefact
`results/dw_algo.d4814f37.json` (reconstructed from transcript).

Sweep of algos {-1, 0..23} on head (B=16352, N=50272, K=1024) and fc
(B=16352, N=4096, K=1024), 10 interleaved trials each vs default:

- head: default (-1) **74.3 ms** — the FASTEST; explicit algos 335–425+ ms
  (4.5–5.7× slower). Parity vs default: max|Δ| = 0 for algo -1, ≤ 2.4e-4 for
  others (sign-disagreement gate would have allowed them, but they lose).
- fc: default 5.43 ms, again fastest.

**Conclusion: cuBLAS's heuristic already picks the optimal algo on T4 for
these shapes. Item #7 closed as measured-negative; branch not merged.**
(The pytest rc=1 in the session was the artefact gate flagging the
unprovenanced scale_fold json on the branch — fixed on master @ 3ded021 —
not a code failure.)

Note for the record: dW head = 74 ms vs fwd head ~51 ms at equal FLOPs is a
cuBLAS-internal property of the [K,N,B] skinny-M layout, not a mis-set algo.
