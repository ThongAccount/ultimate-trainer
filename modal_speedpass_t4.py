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
    .uv_pip_install("uv", "torch", "ninja", "huggingface_hub", "numpy", "setuptools", "pytest")
    .apt_install("git", "git-lfs", "cmake", "ninja-build")
    .run_commands("git lfs install && git lfs install --system")
)

# Don't modify @app.function imports, especially GPU, CPU and Memory
@app.function(
    image=image,
    gpu="T4",
    cpu=2,
    memory=4 * 1024,
    timeout=1800,
)
def speedpass_benchmark(phase: str = "all"):
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

    # ── Environment info ──
    print("\n--- Environment ---")
    print("GPU:", torch.cuda.get_device_name(0))
    print("Compute capability:", torch.cuda.get_device_capability(0))
    print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2), "GiB")
    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())

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
        "gpu": torch.cuda.get_device_name(0),
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

    # ── Phase: SubQSA correctness tests ──
    if phase in ("all", "subqsa"):
        print("\n" + "=" * 70)
        print("PHASE: SUBQSA CORRECTNESS TESTS")
        print("=" * 70)

        r = subprocess.run(
            [sys.executable, "-m", "pytest",
             "tests/test_subqsa_comprehensive.py",
             "tests/test_subqsa_cuda_integration.py",
             "tests/test_subqsa_selection.py",
             "tests/test_subqsa_window.py",
             "-q"],
            capture_output=True, text=True, timeout=900,
        )
        # Count pass/fail summary line (pytest -q prints "N passed, M failed")
        summary = [l for l in r.stdout.split('\n') if 'passed' in l or 'failed' in l]
        for s in summary[-3:]:
            print(f"  {s}")
            results.setdefault("subqsa_tests", []).append(s.strip())
        if r.returncode != 0:
            results["subqsa_tests_status"] = "FAIL"
            print("  TEST STDERR tail:")
            print(r.stderr[-2000:])
        else:
            results["subqsa_tests_status"] = "PASS"

        # ── Phase: SubQSA combine benchmark ──
        print("\n" + "=" * 70)
        print("PHASE: SUBQSA COMBINE BENCHMARK")
        print("=" * 70)

        r = subprocess.run(
            [sys.executable, "bench_subqsa_combine.py", "--iters", "30"],
            capture_output=True, text=True, timeout=900,
        )
        print(r.stdout)
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

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE — SUMMARY")
    print("=" * 70)
    print(json.dumps(results, indent=2))

    with open("/tmp/speedpass_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results