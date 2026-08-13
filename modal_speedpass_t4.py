# This file is a Modal Speedpass Benchmark file.
# Customize the benchmark at def speedpass_benchmark.
# To run it: modal run modal_speedpass_t4.py::speedpass_benchmark

import modal

app = modal.App()

cuda_version = "12.8.1"  # should be no greater than host CUDA version
flavor = "devel"  # includes full CUDA toolkit
operating_sys = "ubuntu24.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .uv_pip_install("uv", "torch", "ninja", "huggingface_hub")
    .apt_install("git", "git-lfs", "cmake", "ninja")
    .run_commands("git lfs install && git lfs install --system")
)

# Don't modify @app.function imports, especially GPU, CPU and Memory
@app.function(
    image=image,
    gpu="T4",
    cpu=2,
    memory=4 * 1024, 
)
def speedpass_benchmark():
    # Write your own benchmark here.
    # Below is an example:
    import os
    import subprocess
    import torch

    # Clone repo if needed, else just comment it.
    result = subprocess.run(["git", "clone", "https://github.com/ThongAccount/ultimate-trainer.git"], capture_output=True, text=True)
    print(result.stdout)

    print("=" * 60)
    print("MODAL T4 SPEEDPASS SANITY CHECK")
    print("=" * 60)

    print("GPU:", torch.cuda.get_device_name(0))
    print("Compute capability:", torch.cuda.get_device_capability(0))
    print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2), "GiB")
    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())

    print("\n--- CUDA smoke test ---")

    x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    y = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)

    # Warmup
    for _ in range(10):
        torch.mm(x, y)

    torch.cuda.synchronize()

    import time
    start = time.perf_counter()

    for _ in range(100):
        torch.mm(x, y)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    print(f"100x 4096x4096 FP16 matmul: {elapsed:.3f}s")
    print(f"Average: {elapsed / 100 * 1000:.3f} ms")

    print("\n--- Environment ---")
    subprocess.run(["nvidia-smi"])

    print("=" * 60)
    print("T4 SANITY CHECK PASSED")
    print("=" * 60)

    
