"""Modal e2e A/B: baseline vs 64x64-dispatch on train_gigatoken, same container.

Mounts local tree (HEAD=b23359b) + baseline files (HEAD~1) at /baseline.
Copies repo to writable /work, swaps files, runs both variants from /work.
"""
import modal

app = modal.App("e2e-ab-dispatch")

cuda_version = "12.8.1"
flavor = "devel"
os_name = "ubuntu24.04"
tag = f"{cuda_version}-{flavor}-{os_name}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .uv_pip_install("uv", "torch", "ninja", "huggingface_hub", "numpy", "setuptools", "pytest", "gigatoken")
    .apt_install("git", "git-lfs", "cmake", "ninja-build")
    .add_local_dir(".", remote_path="/repo", ignore=[".git", ".venv", "__pycache__", "*.pyc", ".pytest_cache"])
    .add_local_dir("/tmp/baseline", remote_path="/baseline")
)

FILES = [
    "kernels/packed_ternary/custom_ops.py",
    "kernels/packed_ternary/pack_update.py",
    "kernels/packed_ternary/gemm_backward_dx_tc.cu",
]


@app.function(image=image, gpu="T4", cpu=2, memory=4 * 1024, timeout=2400)
def run_ab() -> str:
    import subprocess, sys, shutil, os
    import torch
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

    # writable copy of repo
    shutil.copytree("/repo", "/work", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".venv", ".git"))

    subprocess.run([sys.executable, "-c",
        "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt','shakespeare.txt')"],
        cwd="/work", check=True)

    # snapshot HEAD files before any swap
    for f in FILES:
        shutil.copy(f"/work/{f}", f"/work/{f}.head")

    results = {}
    for label, use_baseline in [("HEAD (dispatch fix)", False), ("BASELINE (HEAD~1)", True)]:
        if use_baseline:
            for f in FILES:
                shutil.copy(f"/baseline/{os.path.basename(f)}", f"/work/{f}")
        else:
            for f in FILES:
                shutil.copy(f"/work/{f}.head", f"/work/{f}")

        print(f"\n===== {label} =====", flush=True)
        r = subprocess.run(
            [sys.executable, "train_gigatoken.py", "--text", "shakespeare.txt"],
            cwd="/work", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=900,
        )
        out = r.stdout
        results[label] = out
        print("\n".join(out.splitlines()[-12:]), flush=True)

    print("\n\n===== SUMMARY (last 8 lines each) =====", flush=True)
    for k, v in results.items():
        print(f"--- {k} ---")
        print("\n".join(v.splitlines()[-8:]))
    return "done"


@app.local_entrypoint()
def main():
    out = run_ab.remote()
    print(out)