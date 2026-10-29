# Contributing

## Commit messages — Conventional Commits, scope REQUIRED
Format: `type(scope): subject` — lowercase imperative subject, header ≤ 100 chars.
Allowed types: docs, feat, fix, chore, refactor, perf, test, build, ci.
Enforced on PRs by CI (.github/workflows/commit-lint.yml + commitlint.config.js).
Grandfathering: legacy subjects on master that predate this rule stay untouched —
NO history rewrite; master direct pushes cannot be linted retroactively, so the
rule applies to new commits (PRs, and new direct pushes where lintable).

## Branching & push pattern
Each feature gets its own branch (e.g. `speedpass-foo`); develop and HOLD it on
the branch until it works (CPU tests + the Colab T4 gate pass), then merge to
master. Push pattern: conventional commit on the branch → `git push -u origin <branch>`
→ merge to master → `git push origin master`. Every benchmark record must quote the
git commit (scripts/colab_setup.sh prints it; every colab_* runner embeds it in
RESULTS JSON).

## Read-only audits
Hygiene audits fan out as 5 batches of 4 parallel subagents (20 read-only
investigations), each strictly READ-ONLY (no files created/modified) and each
returning exact, copy-pasteable remediation content (diffs, commands, file
contents) plus a ranked conventional-commit list — never applying changes itself.

## Scratch files
Scratch/ephemeral files go in /workspace/temp, NOT /tmp (VM-side /tmp is wiped
between calls).

## Dependencies
requirements.txt = lower bounds only; requirements-lock.txt = exact Colab T4 image
(regenerate on the VM: `pip freeze | grep -vE '^(-e |git\\+|.*@ file://)' > requirements-lock.txt`).
Kernels JIT against the installed torch ABI; TORCH_CUDA_ARCH_LIST defaults to 7.5
(T4) centrally in kernels/common.py — set the env var to target another GPU.

## Tests
CPU-import-safe tests run in CI (.github/workflows/pytest-cpu.yml); the real
correctness gate is Colab T4 via colab_pytest.py (clones committed HEAD, runs the
kernel suite on GPU). Never merge kernel changes without a green Colab gate.

## Artefacts
Every results/*.json must carry a provenance block — write artefacts only through
tools/artefact_provenance.write_artefact (invalid ones are quarantined to
results/quarantine/ + results/INVALID.md). tests/conftest.py enforces this on all
committed results/*.json at every pytest session (SKIP_ARTEFACT_GATE=1 to bypass).
