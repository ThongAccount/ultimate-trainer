"""Modal: GPU residency experiment (session 9)."""
import modal

app = modal.App("gpu-residency-exp")

tag = "12.8.1-devel-ubuntu24.04"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .uv_pip_install("uv", "torch", "ninja", "huggingface_hub", "numpy", "setuptools", "pytest", "gigatoken")
    .apt_install("git", "git-lfs", "cmake", "ninja-build")
    .add_local_dir(".", remote_path="/repo", ignore=[".git", ".venv", "__pycache__", "*.pyc", ".pytest_cache", "docs", "*.md"])
)


@app.function(image=image, gpu="T4", cpu=2, memory=8 * 1024, timeout=2400)
def run() -> str:
    import subprocess, sys, torch
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    r = subprocess.run([sys.executable, "tests/experiment_gpu_residency.py"],
                       cwd="/repo", stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, timeout=1700)
    print(r.stdout, flush=True)
    return r.stdout


@app.local_entrypoint()
def main():
    out = run.remote()
    with open("/tmp/residency.txt", "w") as f:
        f.write(out)
    print(out)
