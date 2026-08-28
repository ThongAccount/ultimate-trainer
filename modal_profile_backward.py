"""Modal runner: attribute backward-pass time (GPU vs CPU/dispatch)."""
import modal

app = modal.App("bwd-attribution")

cuda_version = "12.8.1"
flavor = "devel"
os_name = "ubuntu24.04"
tag = f"{cuda_version}-{flavor}-{os_name}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .uv_pip_install("uv", "torch", "ninja", "huggingface_hub", "numpy", "setuptools", "pytest")
    .apt_install("git", "git-lfs", "cmake", "ninja-build")
    .add_local_dir(".", remote_path="/repo", ignore=[".git", ".venv", "__pycache__", "*.pyc", ".pytest_cache"])
)

@app.function(image=image, gpu="T4", cpu=2, memory=4*1024, timeout=2400)
def run() -> str:
    import subprocess, sys
    import torch
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("PyTorch:", torch.__version__, flush=True)
    r = subprocess.run([sys.executable, "tests/profile_backward_attribution.py"],
        cwd="/repo", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=1800)
    return r.stdout

@app.local_entrypoint()
def main():
    out = run.remote()
    with open("/tmp/profile_backward.txt", "w") as f:
        f.write(out)
    print(out)