# RUNBOOK 2026-10-31 — CI-built prebuilt CUDA kernels (no on-VM nvcc)

Problem: every `load_inline` kernel compiled with nvcc on the Colab VM at run
start, costing 2-5 min of free T4 session time per extension per fresh VM.
Fix: GitHub Actions builds the whole torch extension cache once per kernel
change and publishes it as a release asset; the VM downloads and extracts it,
and `load_inline` then skips nvcc entirely. The cache hit is ninja
mtime-based in torch 2.11: `load_inline` rewrites main.cpp/cuda.cu only when
their content changes (`_maybe_write`), so with the bundle extracted and the
.so files touched newer than the sources, `ninja` reports "no work to do"
and the prebuilt .so is dlopened (the bundle MUST therefore include the .o
and .ninja_deps files — without them ninja recompiles everything).

## Pipeline

- `.github/workflows/build-kernels.yml` triggers on pushes to `master` that
  touch `kernels/**`, `build_kernels.py`, `tools/build_kernel_bundle.py`,
  `tools/fetch_prebuilt.py`, `requirements.txt`, or the workflow itself
  (plus `workflow_dispatch`).
- Job runs in `nvidia/cuda:13.0.1-devel-ubuntu24.04` with python 3.13 and
  `torch==2.11.0+cu130` — the exact Colab T4 VM stack (provenance of
  `results/*.B05prod.json`: python 3.13.16, torch 2.11.0+cu130, cuda 13.0,
  sm_75). `TORCH_CUDA_ARCH_LIST=7.5` is set by `kernels/common.py`.
  No GPU is needed to compile.
- `tools/build_kernel_bundle.py` imports the full production kernel surface
  and forces every lazy `_load*`/`_ensure_loaded`, building all extensions
  into a scratch `TORCH_EXTENSIONS_DIR`, then tars it (minus .ptx/.cubin
  intermediates only — .o/.ninja_deps/.ninja_log must ship, see above) to
  `dist/kernels-bundle.tar.gz` + `manifest.json`
  (commit, torch, cuda, python, py_tag, nvcc, ext count).
- Assets `kernels-bundle.tar.gz`, `.sha256`, `manifest.json` are uploaded
  (clobber same-name assets) to release tag **`kernels-latest`** (prerelease)
  by `tools/publish_bundle.py` (stdlib only — the CUDA container has no
  curl/gh).

## VM side

- `tools/fetch_prebuilt.py` (`ensure_prebuilt(repo_dir)`) is called by all
  three `scripts/colab_*.py` drivers right after the clone, and by
  `scripts/colab_setup.sh`. It sets `TORCH_EXTENSIONS_DIR` (default
  `~/.cache/torch_extensions`), downloads the three assets via the GitHub
  API with `GITHUB_TOKEN`, verifies sha256, and extracts into the extensions
  dir. Children inherit the env var, so no child JITs.
- Hard checks (any failure -> `PREBUILT: SKIP`, JIT fallback, run continues):
  token present, assets present, sha256 match, `manifest.torch ==
  torch.__version__`, `manifest.py_tag == py{sys.version_info}_cu{...}`,
  no `..`/absolute tar members. Commit mismatch vs the clone HEAD is a WARN
  only (doc-only pushes may post-date the bundle); the `--expect-commit`
  guard pins the clone and the P0 gate bit-reproduction is the end-to-end
  check that the prebuilt kernels match the sources.
- Disable with `UT_PREBUILT=0`; select another tag with `UT_KERNELS_TAG`.
- `TORCH_EXTENSIONS_DIR` and `UT_PREBUILT` are in the artefact provenance
  env whitelist, so every results/*.json records which path was taken.

## If the workflow is red

1. `gh run list --workflow build-kernels` / view the log; common causes:
   runner disk (cuda image + torch cu130 ~ 12-15 GB; ubuntu-24.04 has ~30 GB),
   a kernel source that no longer compiles under nvcc 13.0, pip index drift.
2. The VM side never depends on CI being green: missing release -> JIT
   fallback, same behaviour as before this change.
3. Never hand-edit the `kernels-latest` release; it is fully owned by CI.

## Validating a new bundle

The 300-step production run's P0 phase (synthetic flip gate at B=5) must
bit-reproduce the JIT-built B05x/B05prod values (V 0.00494, M 0.0294,
C 1.035) — that is the proof the prebuilt .so files are equivalent to the
on-VM JIT builds.
