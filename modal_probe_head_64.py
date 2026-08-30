"""Modal runner: probe 64x64 backward dX on HEAD shape (N=50272, non-64-mult)."""
import modal

app = modal.App("probe-head-64")

cuda_version = "12.8.1"
flavor = "devel"
os_name = "ubuntu24.04"
tag = f"{cuda_version}-{flavor}-{os_name}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .uv_pip_install("uv", "torch", "ninja", "numpy", "setuptools")
    .apt_install("git", "cmake", "ninja-build")
    .add_local_dir(".", remote_path="/repo", ignore=[".git", ".venv", "__pycache__", "*.pyc", ".pytest_cache"])
)


@app.function(image=image, gpu="T4", cpu=2, memory=4 * 1024, timeout=2400)
def run_probe() -> str:
    import subprocess, sys, torch
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("Compute:", torch.cuda.get_device_capability(0), flush=True)
    r = subprocess.run(
        [sys.executable, "tests/probe_head_64.py"],
        cwd="/repo", stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=1800,
    )
    return r.stdout


@app.local_entrypoint()
def main():
    out = run_probe.remote()
    with open("probe_head_64_output.txt", "w") as f:
        f.write(out)
    print(out)