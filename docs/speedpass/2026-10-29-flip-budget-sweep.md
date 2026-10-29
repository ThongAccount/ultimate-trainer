# 2026-10-29 — T4 flip-budget sweep runbook (verified against code + colab CLI 0.7.5)

## Status / purpose
Purpose: sweep the per-weight flip budget B in {5,10,20,50} on one T4 per point using
`scripts/colab_flip_rate_gate.py`, to test whether a count-based flip cap breaks the
measured flip-rate limit cycle (gate FAIL at 015763a: results/flip_rate_gate.json rc=1,
M ratio 1.08197 > 0.10). Criteria pre-registered (2026-10-26 V/M/C; 2026-10-29 F0/F1/F2,
commit 0daf735) in tools/flip_rate_verdict.py; the script only measures.
**ON HOLD per user (0.00 CU since 2026-10-21) — do not launch without a go-ahead.**
Verified read-only against the code at 557fbd7/a05ac53; colab CLI 0.7.5 (`colab version`).

## Dead-zone provenance note
The counter dead zone was never measured on T4 in any committed artefact: the only
dead_zone value in results/ is the backfilled label `"dead_zone": 0.0` on the legacy
015763a artefact (results/flip_rate_gate.json, provenance.inferred_from_log=true,
backfilled by fc8fca4/c875efb); its .log has no commit= line and zero
FLIPRATE_BUDGET_VERDICT lines. The "ruled out empirically" claim traces only to the
0daf735 commit message and the comment at tools/flip_rate_verdict.py:84-87 —
pre-registered analysis, no committed T4 dead-zone run. This sweep therefore runs with
--dead-zone 0.0 (default) and varies only --flip-budget.

## Pre-flight session sweep
```
colab version                 # -> Version: 0.7.5   (colab --version does NOT exist in 0.7.5)
colab usage                   # CU balance; 0.00 CU => T4 allocation answers 503 Service Unavailable
colab sessions                # list-only in 0.7.5 — there is NO `colab sessions cleanup`
colab stop -s ut-fb-10        # the single stale EXPIRED entry (token_expires_at 2026-10-29T05:47:36Z, running: null)
```
Never reuse a session name (ut-fb-10 was the stale-clone / driver-skew root cause).
Session limit is 1 (TooManyAssignmentsError / "Allocation refused").

## Exact commands (SHA placeholder)
Let `<SHA>` = the master HEAD at launch time (currently a05ac53…; prefix ok).
```
# 0) smoke — plumbing only; prints FLIPRATE_GATE: SMOKE and exits 0
colab run --gpu T4 -s ut-fbs-smoke --keep --timeout 900 \
  scripts/colab_flip_rate_gate.py --smoke --expect-commit <SHA>
colab log -s ut-fbs-smoke -n 40
colab stop -s ut-fbs-smoke

# 1) one sweep point per budget B in {5,10,20,50} — FRESH session name each time
colab run --gpu T4 -s ut-fbs-05 --keep --timeout 3600 \
  scripts/colab_flip_rate_gate.py \
  --steps 1000 --threshold 32 --backends custom,cublas \
  --dead-zone 0.0 --flip-budget 5 --expect-commit <SHA>
# (private clone: add --env GITHUB_TOKEN="$GITHUB_TOKEN" as a run option; value never printed)

# 2) collect artefacts, then release the VM
colab ls -s ut-fbs-05 /content/ultimate-trainer/results
colab log -s ut-fbs-05 -n 40
colab download -s ut-fbs-05 /content/ultimate-trainer/results/flip_rate_gate.json results/flip_rate_gate.B05.json
colab download -s ut-fbs-05 /content/ultimate-trainer/results/flip_rate_gate.log  results/flip_rate_gate.B05.log
colab stop -s ut-fbs-05
```
Notes: everything after the script path is forwarded verbatim to the script
(`colab run --help`). `--timeout` is float seconds with default **30.0** — always pass it.
The script clones and `git reset --hard origin/master` on the VM; `--expect-commit`
refuses to run (rc 2) if clone HEAD != <SHA>. `--keep` keeps the VM until `colab stop`.

## Expected output lines
Host stdout: `commit=<full 40-char HEAD sha>` first (since the writer adoption), then relayed driver lines, then
`ARTEFACT: /content/ultimate-trainer/results/flip_rate_gate.json`.
Per arm (main, control; plus main_base when --flip-budget>0):
`FLIPRATE: arm=... update_backend=... steps=1000 threshold=32 weights=262144 noisy_rate=... clean_rate=... resid_noisy=... resid_clean=...`
Per 100-step bucket: `FLIPRATE_BUCKET: arm=... update_backend=... step=100..1000 flips_per_step=... flip_rate=... resid_rms=...`
Per backend: three `FLIPRATE_CHECK: update_backend=... V_non_vacuity|M_decay_when_noise_removed|C_control_sustains_flips ok=... value=... threshold=... detail=...`
and `FLIPRATE_VERDICT: update_backend=... passed=...`; when --flip-budget>0 additionally
the main_base block, three F-checks (F0_non_vacuity, F1_budget_reduces_flips,
F2_budgeted_meets_decay) and `FLIPRATE_BUDGET_VERDICT: update_backend=... flip_budget=<B> passed=...`.
Final: `FLIPRATE_GATE: PASS|FAIL (shipped backend = cublas)` (last entry of --backends);
with --smoke also a `FLIPRATE_GATE: SMOKE (criteria not evaluated)` line — **a log parser
must take the LAST gate line / rc, not the first** (smoke prints FAIL then SMOKE, exits 0).
Mismatch path: `COMMIT-MISMATCH: clone HEAD=... does not match --expect-commit=...` then exit 2.

## Exit-code table
| rc | meaning | artefact written? |
|----|---------|-------------------|
| 0 | gate PASS (V+M+C on shipped backend), or --smoke | yes, rc:0 |
| 1 | FLIPRATE_GATE: FAIL — a RESULT, not a crash; also unhandled driver crash | yes rc:1 on gate FAIL; no if crash before artefact write |
| 2 | COMMIT-MISMATCH only — BEFORE driver write and measurement (argparse unknown-arg exits 2 before any artefact write) | quarantined: results/quarantine/flip_rate_gate.<sha>.json + INVALID.md line (never results/flip_rate_gate.json) |

## Artefact schema
results/flip_rate_gate.json: {"commit" (full 40-char), "provenance", "steps", "threshold",
"backends", "dead_zone", "flip_budget", "rc", "lines"}; .log first line `commit=<full sha>`. Committed in-repo results/*.json carry a provenance
block and are validated by the pytest artefact gate (tests/conftest.py,
SKIP_ARTEFACT_GATE=1 to bypass).

## Capacity policy
- Run `colab usage` before every allocation. At 0.00 CU the T4 allocation answers 503
  Service Unavailable. `colab pay` opens the compute-unit signup page.
- Session limit is 1. ONE allocation at a time, NO polling/retry loops. On refusal:
  ABANDON the attempt, `colab stop -s <name>` any leftover, re-check `colab usage`,
  resume only when balance > 0.

## Verdict interpretation
FLIPRATE_GATE = V+M+C on the LAST backend of --backends only; T=32:
V main noisy >= 0.25/T^2 = 2.4414e-4; M main clean/noisy <= 0.10; C control clean/noisy
>= 0.50 with control noisy > 0 (tools/flip_rate_verdict.py). When --flip-budget>0,
FLIPRATE_BUDGET_VERDICT reports F0 (main_noisy>0 and no-budget clean>0), F1 (budgeted
clean / no-budget clean <= 0.50 — the in-run main_base arm supplies the denominator),
F2 (budgeted clean / noisy <= 0.10) — reported but NOT ANDed into FLIPRATE_GATE.
Pre-registered negative outcome: if no B in {5,10,20,50} satisfies V+M+C together,
the flip budget also fails the gate — a valid pre-registered negative result.

## Post-run checklist
1. `colab log -s <name> -n 40`: first line must be `commit=<full SHA>` matching
   --expect-commit; if COMMIT-MISMATCH appeared, rc was 2 and the only artefact is the
   quarantined results/quarantine/flip_rate_gate.<sha>.json — never commit results/flip_rate_gate.json from an rc=2 run.
2. Download json+log per budget; verify keys incl. flip_budget == B, dead_zone == 0.0,
   rc in {0,1}; .log first line is commit=... .
3. Save each run under a distinct filename (results/flip_rate_gate.B05.json/.log, ...)
   — never overwrite the same path across runs (ut-fb-10 failure mode).
4. `colab stop -s <name>`; `colab sessions` must show no leftover; never reuse the name.
5. Commit artefacts with the measured SHA in the message; the pytest artefact gate
   validates tracked results/*.json provenance.
6. Tabulate FLIPRATE_VERDICT (V/M/C) and FLIPRATE_BUDGET_VERDICT (F0/F1/F2) per B;
   record PASS/FAIL per budget against the pre-registered criteria before any criteria change.

## Known pre-flight fixes already landed (do not re-run the old buggy driver)
- 557fbd7: DRIVER threshold formatter `_th()` — F0's string threshold `"> 0"` no longer
  crashes budgeted runs (bug B1 found by CPU dry-run).
- 07001be/6192a47/b96a6ba/557fbd7/a05ac53: decl arity, vacuity guards, CUDAGraph +
  packed_linear flip threading, negative-budget + flip_count shape/dtype guards.
- First T4 run after these must validate the CUDA compile of the fixed v2 decls
  (no nvcc on CPU boxes).
