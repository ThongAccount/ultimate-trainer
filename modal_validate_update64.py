"""Modal: validate + bench the fixed 64×64 update kernel (local tree mount)."""
import modal

app = modal.App("validate-update64")

tag = "12.8.1-devel-ubuntu24.04"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .uv_pip_install("uv", "torch", "ninja", "huggingface_hub", "numpy", "setuptools", "pytest", "gigatoken")
    .apt_install("git", "git-lfs", "cmake", "ninja-build")
    .add_local_dir(".", remote_path="/repo", ignore=[".git", ".venv", "__pycache__", "*.pyc", ".pytest_cache", "docs", "*.md"])
)


@app.function(image=image, gpu="T4", cpu=2, memory=6 * 1024, timeout=1800)
def run() -> str:
    import subprocess, sys, torch
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    out = []
    for label, cmd in [
        ("CORRECTNESS_64", [sys.executable, "tests/test_update64_dimensional_bug.py"]),
        ("BENCH_64v32", [sys.executable, "tests/bench_update_64_vs_32.py"]),
    ]:
        print(f"\n===== {label} =====", flush=True)
        r = subprocess.run(cmd, cwd="/repo", stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, timeout=1200)
        print(r.stdout, flush=True)
        out.append(f"{label}: rc={r.returncode}\n{r.stdout}")
        if r.returncode != 0 and label == "CORRECTNESS_64":
            break  # don't bench a wrong kernel
    return "\n".join(out)


@app.local_entrypoint()
def main():
    out = run.remote()
    with open("/tmp/validate_update64.txt", "w") as f:
        f.write(out)
    print(out)
