"""
DDP worker for GPU validation (used by test_modal_2xT4.py).
Must be a standalone module so torch.multiprocessing.spawn can pickle it.
"""

import torch
import torch.distributed as dist
import torch.nn as nn
from ultimate_trainer.config import UltimateModelConfig
from ultimate_trainer.model import UltimateModel


def ddp_worker(rank, world_size, mc_kwargs: dict):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    mc = UltimateModelConfig(**mc_kwargs)
    model = UltimateModel(mc).to(device)
    ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    opt = torch.optim.AdamW(ddp_model.parameters(), lr=1e-3)
    ids = torch.randint(0, 4096, (2, 64), device=device)

    for step in range(10):
        opt.zero_grad()
        loss = ddp_model.get_loss(ids)
        loss.backward()
        opt.step()

    if rank == 0:
        print(f"  DDP rank0 final loss: {loss.item():.4f}")

    dist.destroy_process_group()
