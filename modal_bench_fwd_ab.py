"""Modal: A/B bench forward TC64 — baseline (7d07369, K16) vs exp (K32)."""
import modal

app = modal.App("bench-fwd-k32")

tag = "12.8.1-devel-ubuntu24.04"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .uv_pip_install("uv", "torch", "ninja", "huggingface_hub", "numpy", "setuptools", "pytest", "gigatoken")
    .apt_install("git", "git-lfs", "cmake", "ninja-build")
    .add_local_dir("/tmp/uam_fwdbase", remote_path="/repo_base", ignore=[".git", ".venv", "__pycache__", "*.pyc", ".pytest_cache", "docs", "*.md"])
    .add_local_dir(".", remote_path="/repo_exp", ignore=[".git", ".venv", "__pycache__", "*.pyc", ".pytest_cache", "docs", "*.md"])
)


@app.function(image=image, gpu="T4", cpu=2, memory=8 * 1024, timeout=2400)
def run(repeats: int = 3, do_probe: bool = True) -> str:
    import subprocess, sys, torch
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    lines = []
    if do_probe:
        for label, repo in [("BASE", "/repo_base"), ("EXP", "/repo_exp")]:
            r = subprocess.run([sys.executable, "tests/test_gemm_forward.py", "-q"], cwd=repo,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, timeout=900)
            ok = "passed" in r.stdout and "failed" not in r.stdout
            print(f"--- FWD PARITY {label} rc={r.returncode} ok={ok} ---\n{r.stdout[-1200:]}", flush=True)
            lines.append(f"FWD_PARITY {label}: ok={ok}")
            if not ok:
                return "\n".join(lines)
    for trial in range(repeats):
        for label, repo in [("BASE", "/repo_base"), ("EXP", "/repo_exp")]:
            r = subprocess.run([sys.executable, "tests/bench_fwd_k.py"], cwd=repo,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, timeout=900)
            for l in r.stdout.splitlines():
                if ("fc" in l or "head" in l) and "in=" in l:
                    print(f"trial{trial} {label}: {l}", flush=True)
                    lines.append(f"trial{trial} {label}: {l}")
    return "\n".join(lines)


@app.local_entrypoint()
def main(repeats: int = 3):
    out = run.remote(repeats)
    print("==== SUMMARY ====")
    print(out)