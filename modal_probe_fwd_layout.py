"""Modal: in-process paired A/B of forward W_smem layout (old transposed vs new row-major).

Mounts local tree so the modified kernel + probe ship together; the probe
compiles both variants from source inside the runner.
"""
import modal

app = modal.App("fwd-layout-ab")

cuda_version = "12.8.1"
flavor = "devel"
os_name = "ubuntu24.04"
tag = f"{cuda_version}-{flavor}-{os_name}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .uv_pip_install("torch", "ninja", "numpy", "setuptools")
    .apt_install("git", "cmake", "ninja-build")
    .add_local_dir(".", remote_path="/repo", ignore=[".git", ".venv", "__pycache__", "*.pyc", ".pytest_cache", "docs", "*.md", "checkpoints"])
)


@app.function(image=image, gpu="T4", cpu=2, memory=4 * 1024, timeout=2400)
def run() -> str:
    import subprocess, sys
    import torch
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("torch:", torch.__version__, "cuda:", torch.version.cuda, flush=True)
    proc = subprocess.Popen(
        [sys.executable, "tests/probe_fwd_layout.py"],
        cwd="/repo", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    out_lines = []
    for line in proc.stdout:
        line = line.rstrip()
        print(f"  {line}", flush=True)
        out_lines.append(line)
    proc.wait()
    rc = proc.returncode
    out = "\n".join(out_lines)
    if rc != 0:
        return out + f"\n---EXIT {rc}---"
    return out


@app.local_entrypoint()
def main():
    out = run.remote()
    print(out)