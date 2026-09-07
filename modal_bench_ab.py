"""Modal: A/B update_tc_v2 bench — baseline vs dedup, 3 trials each."""
import modal

app = modal.App("bench-update-ab2")

tag = "12.8.1-devel-ubuntu24.04"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .uv_pip_install("uv", "torch", "ninja", "huggingface_hub", "numpy", "setuptools", "pytest", "gigatoken")
    .apt_install("git", "git-lfs", "cmake", "ninja-build")
    .add_local_dir("/tmp/uam_base", remote_path="/repo_base", ignore=[".git", ".venv", "__pycache__", "*.pyc", ".pytest_cache", "docs", "*.md"])
    .add_local_dir(".", remote_path="/repo_dedup", ignore=[".git", ".venv", "__pycache__", "*.pyc", ".pytest_cache", "docs", "*.md"])
)


@app.function(image=image, gpu="T4", cpu=2, memory=4 * 1024, timeout=1800)
def run(repeats: int = 3) -> str:
    import subprocess, sys, torch
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    lines = []
    for trial in range(repeats):
        for label, repo in [("BASE", "/repo_base"), ("DEDUP", "/repo_dedup")]:
            r = subprocess.run([sys.executable, "tests/bench_update_v2.py"],
                               cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, timeout=900)
            ms = [l for l in r.stdout.splitlines() if ("ms" in l and ("fc" in l or "head" in l))]
            for l in ms:
                print(f"trial{trial} {label}: {l}", flush=True)
                lines.append(f"trial{trial} {label}: {l}")
    return "\n".join(lines)


@app.local_entrypoint()
def main(repeats: int = 3):
    out = run.remote(repeats)
    print("==== SUMMARY ====")
    print(out)
