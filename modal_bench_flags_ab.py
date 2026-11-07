"""Modal: A/B bench compiler flags — baseline (-O2) vs exp (-O3 --use_fast_math)."""
import modal

app = modal.App("bench-flags")

tag = "12.8.1-devel-ubuntu24.04"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .uv_pip_install("uv", "torch", "ninja", "huggingface_hub", "numpy", "setuptools", "pytest", "gigatoken")
    .apt_install("git", "git-lfs", "cmake", "ninja-build")
    .add_local_dir("/tmp/uam_flagsbase", remote_path="/repo_base", ignore=[".git", ".venv", "__pycache__", "*.pyc", ".pytest_cache", "docs", "*.md"])
    .add_local_dir(".", remote_path="/repo_exp", ignore=[".git", ".venv", "__pycache__", "*.pyc", ".pytest_cache", "docs", "*.md"])
)

BENCHES = ["tests/bench_dx_k.py", "tests/bench_update_v2.py"]


@app.function(image=image, gpu="T4", cpu=2, memory=8 * 1024, timeout=2400)
def run(repeats: int = 3, do_probe: bool = True) -> str:
    import subprocess, sys, torch
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    lines = []
    if do_probe:
        for label, repo in [("BASE", "/repo_base"), ("EXP", "/repo_exp")]:
            r = subprocess.run([sys.executable, "tests/probe_bwd.py"], cwd=repo,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, timeout=900)
            ok = "PASS" in r.stdout or "err vs torch" in r.stdout
            print(f"--- PROBE {label} rc={r.returncode} ---\n{r.stdout[-600:]}", flush=True)
            lines.append(f"PROBE {label}: rc={r.returncode}")
            if r.returncode != 0:
                return "\n".join(lines)
    for trial in range(repeats):
        for label, repo in [("BASE", "/repo_base"), ("EXP", "/repo_exp")]:
            for b in BENCHES:
                r = subprocess.run([sys.executable, b], cwd=repo,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, timeout=900)
                for l in r.stdout.splitlines():
                    if ("fc" in l or "head" in l) and "in=" in l:
                        print(f"trial{trial} {label} {b.split('/')[-1]}: {l}", flush=True)
                        lines.append(f"trial{trial} {label} {b.split('/')[-1]}: {l}")
    return "\n".join(lines)


@app.local_entrypoint()
def main(repeats: int = 3):
    out = run.remote(repeats)
    print("==== SUMMARY ====")
    print(out)