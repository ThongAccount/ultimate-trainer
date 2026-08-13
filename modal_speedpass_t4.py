# This file is a Modal Speedpass Benchmark file.
# Usage: modal run modal_speedpass_t4.py::speedpass_benchmark

import modal

app = modal.App()

cuda_version = "12.8.1"  # should be no greater than host CUDA version
flavor = "devel"  # includes full CUDA toolkit
operating_sys = "ubuntu24.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .uv_pip_install("uv", "torch", "ninja", "huggingface_hub", "numpy", "setuptools")
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
def speedpass_benchmark():
    import os
    import subprocess
    import torch
    import json
    import time

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

    subprocess.run(["nvidia-smi"], capture_output=False)

    # ── Clone or pull repo ──
    if not os.path.exists(REPO_DIR):
        print(f"\n--- Cloning {REPO_URL} ---")
        subprocess.run(["git", "clone", REPO_URL], check=True)
    else:
        print(f"\n--- Repo exists, pulling ---")

    os.chdir(REPO_DIR)
    subprocess.run(["git", "fetch", "origin"], check=True)
    subprocess.run(["git", "checkout", BRANCH], check=True)
    subprocess.run(["git", "pull", "origin", BRANCH], check=True)

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    print(f"Git commit: {commit}")

    # ── Download shakespeare.txt ──
    if not os.path.exists("shakespeare.txt"):
        import urllib.request
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, "shakespeare.txt")
        print("Downloaded shakespeare.txt")

    # ── Install gigatoken ──
    subprocess.run(["pip", "install", "gigatoken"], capture_output=True)

    # ── Run baseline benchmark ──
    print("\n" + "=" * 70)
    print("PHASE 2: BASELINE BENCHMARK")
    print("=" * 70)

    results = {
        "commit": commit,
        "gpu": torch.cuda.get_device_name(0),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
    }

    # Run train_gigatoken.py
    print("\n--- Running train_gigatoken.py (50 steps) ---")
    result = subprocess.run(
        ["python", "train_gigatoken.py", "--text", "shakespeare.txt"],
        capture_output=True, text=True, timeout=600
    )
    print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
    if result.stderr:
        # Filter out JIT compilation noise
        for line in result.stderr.split('\n')[-20:]:
            if line.strip() and 'warning' not in line.lower():
                print(f"  STDERR: {line}")

    # Parse results
    for line in result.stdout.split('\n'):
        if 'Avg:' in line:
            print(f"\n  BASELINE: {line.strip()}")
            results["baseline_line"] = line.strip()
            # Extract tok/s
            parts = line.split()
            for i, p in enumerate(parts):
                if 'tok/s' in p and i > 0:
                    tok_str = parts[i-1].replace(',', '')
                    try:
                        results["baseline_tok_s"] = float(tok_str)
                    except:
                        pass

    # Save baseline results
    with open("/tmp/baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBaseline results saved: {json.dumps(results, indent=2)}")

    # ── Run bench_parity.py if it exists ──
    if os.path.exists("bench_parity.py"):
        print("\n--- Running bench_parity.py ---")
        try:
            result = subprocess.run(
                ["python", "bench_parity.py"],
                capture_output=True, text=True, timeout=120
            )
            print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
            results["bench_parity"] = result.stdout
        except Exception as e:
            print(f"  bench_parity failed: {e}")

    print("\n" + "=" * 70)
    print("BASELINE BENCHMARK COMPLETE")
    print(f"Results: {json.dumps(results, indent=2)}")
    print("=" * 70)

    return results
