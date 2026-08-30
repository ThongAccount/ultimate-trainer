"""Modal runner: full pytest correctness suite on GPU (T4)."""
import modal

app = modal.App("pytest-suite")

cuda_version = "12.8.1"; flavor = "devel"; os_name = "ubuntu24.04"
tag = f"{cuda_version}-{flavor}-{os_name}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .uv_pip_install("uv", "torch", "ninja", "numpy", "setuptools", "pytest")
    .apt_install("git", "cmake", "ninja-build")
    .add_local_dir(".", remote_path="/repo", ignore=[".git", ".venv", "__pycache__", "*.pyc", ".pytest_cache"])
)

@app.function(image=image, gpu="T4", cpu=2, memory=4 * 1024, timeout=2400)
def run() -> str:
    import subprocess, sys, torch
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_packed_linear.py",
         "tests/test_gemm_update.py", "tests/test_kernels.py",
         "tests/test_packed_ternary.py", "-q",
         "--no-header", "--tb=short"],
        cwd="/repo", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=1800)
    return r.stdout

@app.local_entrypoint()
def main():
    out = run.remote()
    open("pytest_suite_output.txt", "w").write(out)
    print(out)