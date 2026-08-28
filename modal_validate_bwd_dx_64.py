"""Modal runner: validate + benchmark 64x64 backward dX kernel on T4.

Mounts the LOCAL repo so uncommitted changes (kernel fix + dispatch routing)
run as-is. Streams results back.
"""
import modal

app = modal.App("validate-bwd-dx-64")

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


@app.function(image=image, gpu="T4", cpu=2, memory=4 * 1024, timeout=2400)
def run_validate() -> str:
    import subprocess, sys, torch
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("Compute:", torch.cuda.get_device_capability(0), flush=True)
    print("PyTorch:", torch.__version__, flush=True)
    r = subprocess.run(
        [sys.executable, "tests/validate_bwd_dx_64.py"],
        cwd="/repo", stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=1800,
    )
    return r.stdout


@app.local_entrypoint()
def main():
    out = run_validate.remote()
    with open("validate_bwd_dx_64_output.txt", "w") as f:
        f.write(out)
    print(out)