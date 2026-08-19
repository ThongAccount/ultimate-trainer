# This file is a Modal Speedpass Benchmark file.
# Usage: modal run modal_speedpass_t4.py::speedpass_benchmark [--phase gigatoken|subqsa|all]

import modal

app = modal.App()

cuda_version = "12.8.1"  # should be no greater than host CUDA version
flavor = "devel"  # includes full CUDA toolkit
operating_sys = "ubuntu24.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .uv_pip_install("uv", "torch", "ninja", "huggingface_hub", "numpy", "setuptools", "pytest", "gigatoken")
    .apt_install("git", "git-lfs", "cmake", "ninja-build")
    .run_commands("git lfs install && git lfs install --system")
)

# Don't modify @app.function imports, especially GPU, CPU and Memory
@app.function(
    image=image,
    gpu="T4",
    cpu=2,
    memory=4 * 1024,
    timeout=2400,
)
def speedpass_benchmark(phase: str = "all", use_gpu: bool = False):
    import os
    import subprocess
    import torch
    import json
    import sys

    REPO_URL = "https://github.com/ThongAccount/ultimate-trainer.git"
    REPO_DIR = "ultimate-trainer"
    BRANCH = "chore/speedpass"

    print("=" * 70)
    print("MODAL T4 SPEEDPASS BENCHMARK")
    print("=" * 70)

    _has_gpu = torch.cuda.is_available()

    # ── Environment info ──
    print("\n--- Environment ---")
    if _has_gpu:
        print("GPU:", torch.cuda.get_device_name(0))
        print("Compute capability:", torch.cuda.get_device_capability(0))
        print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2), "GiB")
    else:
        print("GPU: none (CPU-only)")
    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("CUDA available:", _has_gpu)

    # ── Clone or pull repo ──
    os.chdir("/tmp")
    if not os.path.exists(REPO_DIR):
        print(f"\n--- Cloning {REPO_URL} ---")
        subprocess.run(["git", "clone", REPO_URL], check=True)
    os.chdir(REPO_DIR)
    subprocess.run(["git", "fetch", "origin"], check=True)
    subprocess.run(["git", "checkout", BRANCH], check=True)
    subprocess.run(["git", "pull", "origin", BRANCH], check=True)

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    print(f"Git commit: {commit}")

    results = {
        "commit": commit,
        "gpu": torch.cuda.get_device_name(0) if _has_gpu else "none",
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "phase": phase,
    }

    # ── Download shakespeare.txt ──
    if not os.path.exists("shakespeare.txt"):
        import urllib.request
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, "shakespeare.txt")
        print("Downloaded shakespeare.txt")

    # ── Phase: gigatoken training benchmark ──
    if phase in ("all", "gigatoken"):
        print("\n" + "=" * 70)
        print("PHASE: GIGATOKEN TRAINING BENCHMARK")
        print("=" * 70)

        if not _has_gpu:
            print("  SKIP — no GPU available (CPU-only mode)")
            results["gigatoken_status"] = "SKIP_NO_GPU"
        else:
            proc = subprocess.Popen(
                [sys.executable, "train_gigatoken.py", "--text", "shakespeare.txt"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            stdout_lines = []
            for line in proc.stdout:
                line = line.rstrip()
                print(f"  {line}", flush=True)
                stdout_lines.append(line)
            proc.wait()
            print(f"train_gigatoken exit code: {proc.returncode}")

            for line in stdout_lines:
                if 'Avg:' in line:
                    results["gigatoken_avg"] = line.strip()
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if 'tok/s' in p and i > 0:
                            try:
                                results["gigatoken_tok_s"] = float(parts[i - 1].replace(',', ''))
                            except ValueError:
                                pass
                if '[TIME]' in line:
                    results.setdefault("gigatoken_profile", []).append(line.strip())

    # ── Phase: SubQSA kernel smoke (isolated, streaming; no pytest) ──
    if phase in ("kernel-smoke",):
        print("\n" + "=" * 70)
        print("PHASE: KERNEL SMOKE (fused + sparse parity, streaming)")
        print("=" * 70)
        code = r'''
import torch, sys, os
sys.path.insert(0, "/tmp/ultimate-trainer")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
print("importing kernels...", flush=True)
torch.manual_seed(0)

# 1) Fused combine kernel (dense)
from kernels.subqsa_combine.subqsa_combine import _HAS_SUBQSA_COMBINE, SubQSACombineFn
print("fused available:", _HAS_SUBQSA_COMBINE, flush=True)
B, T, H, D = 2, 64, 2, 32
DO = H * D
x = torch.randn(B, T, DO, device="cuda")
o_cmp = torch.randn(B, H, T, D, device="cuda")
o_slc = torch.randn(B, H, T, D, device="cuda")
o_win = torch.randn(B, H, T, D, device="cuda")
gw1 = torch.randn(64, DO, device="cuda") * 0.1
gw2 = torch.randn(3 * H, 64, device="cuda") * 0.1
onw = torch.randn(DO, device="cuda")
opw = torch.randn(DO, DO, device="cuda") * 0.05
gamma = 0.1

from kernels.subqsa_combine.subqsa_combine import _subqsa_combine_eager
ref = _subqsa_combine_eager(x, o_cmp, o_slc, o_win, gw1, gw2, onw, opw, gamma, None)
print("eager ref done", ref.shape, flush=True)
if _HAS_SUBQSA_COMBINE:
    y = SubQSACombineFn.apply(x, o_cmp, o_slc, o_win, gw1, gw2, onw, opw, gamma, None)
    torch.cuda.synchronize()
    err = (y - ref).abs().max().item()
    print("fused parity err:", err, flush=True)

# 2) Sparse ternary matmul kernel
from kernels.block_sparse_ternary.block_sparse_ternary import block_sparse_ternary_matmul, compute_block_mask, _HAS_CUDA
print("sparse _HAS_CUDA:", _HAS_CUDA, flush=True)
K = DO
num_n = (DO + 63) // 64
num_k = (K + 15) // 16
mask = compute_block_mask(torch.tensor([[0, 1]], device="cuda"), 2, 64, num_n, num_k)
print("mask:", mask.tolist(), "num_n", num_n, "num_k", num_k, flush=True)
xm = torch.randn(B * T, DO, device="cuda")
y_sp = block_sparse_ternary_matmul(xm, opw, gamma, mask)
torch.cuda.synchronize()
wq = torch.clamp(torch.round(opw / gamma), -1, 1) * gamma
ref_mm = xm @ wq.t()
for tn in range(num_n):
    for tk in range(num_k):
        bit = tn * num_k + tk
        if not (mask[bit // 64] & (1 << (bit % 64))):
            ref_mm[:, tn * 64:(tn + 1) * 64] = 0.0
print("sparse parity err:", (y_sp - ref_mm).abs().max().item(), flush=True)
print("SMOKE OK", flush=True)
'''
        r = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=900,
        )
        print(r.stdout, flush=True)
        if r.returncode == 0:
            results["kernel_smoke"] = "PASS"
            print("KERNEL SMOKE PASSED", flush=True)
        else:
            results["kernel_smoke"] = "FAIL"
            print("KERNEL SMOKE FAILED stderr:", flush=True)
            print(r.stderr[-3000:], flush=True)
    if phase in ("all", "subqsa"):
        print("\n" + "=" * 70)
        print("PHASE: SUBQSA CORRECTNESS TESTS")
        print("=" * 70)

        test_files = [
            "tests/test_subqsa_comprehensive.py",
            "tests/test_subqsa_cuda_integration.py",
            "tests/test_subqsa_selection.py",
            "tests/test_subqsa_window.py",
        ]
        if _has_gpu:
            test_files.append("tests/test_speedpass_kernels.py")
        pytest_args = [sys.executable, "-m", "pytest"] + test_files + ["-q"]
        pytest_log = open("/tmp/pytest_out.log", "w")
        p = subprocess.Popen(
            pytest_args,
            stdout=pytest_log, stderr=subprocess.STDOUT, text=True,
        )
        try:
            p.wait(timeout=600)
            rc = p.returncode
            pytest_log.flush()
            r_stdout = open("/tmp/pytest_out.log").read()
        except subprocess.TimeoutExpired:
            p.kill()
            pytest_log.flush()
            r_stdout = open("/tmp/pytest_out.log").read()
            print("  PYTEST TIMEOUT — tail of output so far:", flush=True)
            print(r_stdout[-4000:], flush=True)
            raise RuntimeError("pytest timed out (600s). See tail above.")
        finally:
            pytest_log.close()
        rc = p.returncode
        # Count pass/fail summary line (pytest -q prints "N passed, M failed")
        summary = [l for l in r_stdout.split('\n') if 'passed' in l or 'failed' in l]
        for s in summary[-3:]:
            print(f"  {s}", flush=True)
            results.setdefault("subqsa_tests", []).append(s.strip())
        if rc != 0:
            results["subqsa_tests_status"] = "FAIL"
            # Full pytest output on failure (short tracebacks, no capture)
            print("  PYTEST STDOUT (failures):", flush=True)
            for line in r_stdout.split('\n'):
                if line.strip() and ("_ test" in line or "Error" in line or
                                     "assert" in line or "FAILED" in line or
                                     "raise" in line or "line " in line):
                    print(f"    {line}", flush=True)
            print("  TEST STDERR (from combined stdout+stderr log):", flush=True)
            print(r_stdout[-2000:], flush=True)
        else:
            results["subqsa_tests_status"] = "PASS"

        # ── Phase: SubQSA combine benchmark ──
        print("\n" + "=" * 70)
        print("PHASE: SUBQSA COMBINE BENCHMARK")
        print("=" * 70)

        if not _has_gpu:
            print("  SKIP — no GPU available (CPU-only mode)")
            results["subqsa_combine_status"] = "SKIP_NO_GPU"
        else:
            # Quick parity self-test first (blocking launches catch OOB fast)
            r = subprocess.run(
                [sys.executable, "bench_subqsa_combine.py", "--fast"],
                capture_output=True, text=True, timeout=600,
            )
            print(r.stdout, flush=True)
            if r.returncode != 0:
                print("  SELFTEST FAILED, stderr:", flush=True)
                print(r.stderr[-2000:], flush=True)
                results["subqsa_combine_status"] = "SELFTEST_FAIL"
            else:
                results["subqsa_selftest"] = "PASS"

            # Timed benchmark
            r = subprocess.run(
                [sys.executable, "bench_subqsa_combine.py", "--iters", "15", "--warmup", "5"],
                capture_output=True, text=True, timeout=900,
            )
            print(r.stdout, flush=True)
            bench_lines = []
            for line in r.stdout.split('\n'):
                if 'eager' in line or 'fused' in line:
                    bench_lines.append(line.strip())
                    results.setdefault("subqsa_combine", []).append(line.strip())
            if r.returncode != 0:
                print("  BENCH STDERR tail:", r.stderr[-2000:])
                results["subqsa_combine_status"] = "FAIL"
            else:
                results["subqsa_combine_status"] = "PASS"

    # ── Phase: per-kernel backward profile ──
    if phase in ("all", "profile"):
        print("\n" + "=" * 70)
        print("PHASE: GIGATOKEN BACKWARD PROFILE")
        print("=" * 70)
        if not _has_gpu:
            print("  SKIP — no GPU available")
            results["profile_status"] = "SKIP_NO_GPU"
        else:
            r = subprocess.run(
                [sys.executable, "tests/profile_gigatoken.py"],
                capture_output=True, text=True, timeout=900,
            )
            print(r.stdout, flush=True)
            if r.returncode != 0:
                print("  PROFILE STDERR:", flush=True)
                print(r.stderr[-2000:], flush=True)
                results["profile_status"] = "FAIL"
            else:
                results["profile_status"] = "PASS"

    # ── Phase: bwd_dx correctness probe ──
    if phase in ("all", "bwdprobe"):
        print("\n" + "=" * 70)
        print("PHASE: BWD_DX CORRECTNESS PROBE")
        print("=" * 70)
        if not _has_gpu:
            print("  SKIP — no GPU available")
            results["bwdprobe_status"] = "SKIP_NO_GPU"
        else:
            r = subprocess.run(
                [sys.executable, "tests/probe_bwd.py"],
                capture_output=True, text=True, timeout=600,
            )
            print(r.stdout, flush=True)
            if r.returncode != 0:
                print("  PROBE STDERR:", flush=True)
                print(r.stderr[-1500:], flush=True)
                results["bwdprobe_status"] = "FAIL"
            else:
                results["bwdprobe_status"] = "PASS"

    # ── Phase: 32x64 backward_dx correctness test ──
    if phase in ("all", "test32x64"):
        print("\n" + "=" * 70)
        print("PHASE: 32×64 BACKWARD_DX CORRECTNESS TEST")
        print("=" * 70)
        if not _has_gpu:
            print("  SKIP — no GPU available")
            results["test32x64_status"] = "SKIP_NO_GPU"
        else:
            r = subprocess.run(
                [sys.executable, "tests/test_32x64_correctness.py"],
                capture_output=True, text=True, timeout=600,
            )
            print(r.stdout, flush=True)
            if r.returncode != 0:
                print("  TEST STDERR:", flush=True)
                print(r.stderr[-1500:], flush=True)
                results["test32x64_status"] = "FAIL"
            else:
                results["test32x64_status"] = "PASS"

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE — SUMMARY")
    print("=" * 70)
    print(json.dumps(results, indent=2))

    with open("/tmp/speedpass_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Don't return results — Modal's local deserialization requires torch.
    # The JSON dump + stdout is the source of truth.
    return