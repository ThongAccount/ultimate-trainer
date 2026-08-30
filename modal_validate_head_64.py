"""Modal runner: validate relaxed N_out gate (head dX -> 64x64)."""
import modal

app = modal.App("validate-head-64-dispatch")

cuda_version = "12.8.1"; flavor = "devel"; os_name = "ubuntu24.04"
tag = f"{cuda_version}-{flavor}-{os_name}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .uv_pip_install("uv", "torch", "ninja", "numpy", "setuptools")
    .apt_install("git", "cmake", "ninja-build")
    .add_local_dir(".", remote_path="/repo", ignore=[".git", ".venv", "__pycache__", "*.pyc", ".pytest_cache"])
)

@app.function(image=image, gpu="T4", cpu=2, memory=4 * 1024, timeout=2400)
def run() -> str:
    import subprocess, sys, torch
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    r = subprocess.run([sys.executable, "tests/validate_head_64_dispatch.py"],
        cwd="/repo", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=1800)
    return r.stdout

@app.local_entrypoint()
def main():
    out = run.remote()
    open("validate_head_64_dispatch_output.txt", "w").write(out)
    print(out)