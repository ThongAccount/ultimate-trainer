"""Modal: bench update_tc_v2 at BASELINE (d23f16c, pre-dedup) for A/B."""
import modal

app = modal.App("bench-update-base")

tag = "12.8.1-devel-ubuntu24.04"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .uv_pip_install("uv", "torch", "ninja", "huggingface_hub", "numpy", "setuptools", "pytest", "gigatoken")
    .apt_install("git", "git-lfs", "cmake", "ninja-build")
    .add_local_dir("/tmp/uam_base", remote_path="/repo", ignore=[".git", ".venv", "__pycache__", "*.pyc", ".pytest_cache", "docs", "*.md"])
)


@app.function(image=image, gpu="T4", cpu=2, memory=4 * 1024, timeout=1200)
def run() -> str:
    import subprocess, sys, torch
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    r = subprocess.run([sys.executable, "tests/bench_update_v2.py"], cwd="/repo",
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=900)
    return r.stdout


@app.local_entrypoint()
def main():
    print(run.remote())
