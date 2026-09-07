"""Modal: validate the dedup'd update_tc_v2 kernel (local tree mount).

Runs updatebug correctness, test_gemm_update, and isolated kernel bench.
"""
import modal

app = modal.App("validate-update-dedup")

cuda_version = "12.8.1"
tag = f"{cuda_version}-devel-ubuntu24.04"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .uv_pip_install("uv", "torch", "ninja", "huggingface_hub", "numpy", "setuptools", "pytest", "gigatoken")
    .apt_install("git", "git-lfs", "cmake", "ninja-build")
    .add_local_dir(".", remote_path="/repo", ignore=[".git", ".venv", "__pycache__", "*.pyc", ".pytest_cache", "docs", "*.md"])
)


@app.function(image=image, gpu="T4", cpu=2, memory=4 * 1024, timeout=1800)
def run() -> str:
    import subprocess, sys, torch
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

    out = []
    for label, cmd in [
        ("UPDATEBUG", [sys.executable, "tests/test_update_dimensional_bug.py"]),
        ("GEMM_UPDATE", [sys.executable, "-m", "pytest", "-x", "-q", "tests/test_gemm_update.py"]),
        ("BENCH_UPDATE", [sys.executable, "tests/bench_update_v2.py"]),
    ]:
        print(f"\n===== {label} =====", flush=True)
        r = subprocess.run(cmd, cwd="/repo", capture_output=True, text=True, timeout=900)
        out.append(f"{label}: rc={r.returncode}")
        print(r.stdout[-4000:], flush=True)
        if r.returncode != 0:
            print("STDERR:", r.stderr[-2000:], flush=True)
            out.append(r.stdout[-4000:] + "\n" + r.stderr[-2000:])
            break
    return "\n".join(out)


@app.local_entrypoint()
def main():
    out = run.remote()
    with open("/tmp/validate_update.txt", "w") as f:
        f.write(out)
    print(out)
