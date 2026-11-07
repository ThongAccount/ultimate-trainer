"""Modal: same-instance interleaved e2e A/B of forward layout (old vs new)."""
import modal

app = modal.App("fwd-e2e-ab")

cuda_version = "12.8.1"
flavor = "devel"
os_name = "ubuntu24.04"
tag = f"{cuda_version}-{flavor}-{os_name}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .uv_pip_install("uv", "torch", "ninja", "huggingface_hub", "numpy", "setuptools", "pytest", "gigatoken")
    .apt_install("git", "git-lfs", "cmake", "ninja-build")
    .add_local_dir(".", remote_path="/repo", ignore=[".git", ".venv", "__pycache__", "*.pyc", ".pytest_cache", "docs", "*.md", "checkpoints"])
)


@app.function(image=image, gpu="T4", cpu=2, memory=4 * 1024, timeout=2400)
def run() -> str:
    import subprocess, sys
    import torch
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    proc = subprocess.Popen(
        [sys.executable, "tests/e2e_fwd_ab.py"],
        cwd="/repo", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    out_lines = []
    for line in proc.stdout:
        line = line.rstrip()
        print(f"  {line}", flush=True)
        out_lines.append(line)
    proc.wait()
    return "\n".join(out_lines) + (f"\n---EXIT {proc.returncode}---" if proc.returncode else "")


@app.local_entrypoint()
def main():
    out = run.remote()
    print(out)
