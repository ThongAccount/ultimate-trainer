"""CUDAGraph train step — manual kernel calls, no autograd.

Captures forward + backward_dx into a replayable CUDAGraph.
Update runs outside the graph (due to conditional atomicCAS).

Usage:
    from train_step_graph_v2 import TrainStepGraphCUDAGraph
    graph = TrainStepGraphCUDAGraph(layer, batch_size, in_features, out_features)
    loss = graph.step(x, y_target)  # forward + backward + update
"""

import torch
from kernels.packed_ternary.custom_ops import (
    forward_tc as _forward_op,
    backward_dx_tc as _backward_op,
    update_tc_v2 as _update_op,
)
from kernels.packed_ternary.pack_update import backward_update_fused


class TrainStepGraphCUDAGraph:
    """Captures forward+backward_dx into CUDAGraph. Update runs outside.

    Flow per step:
        1. Graph replay: forward(Y = X @ W^T) + backward_dx(dX = dY @ W)
        2. External: update(W from accumulated gradients)
    """

    def __init__(self, layer: torch.nn.Module, batch_size: int,
                 in_features: int, out_features: int,
                 use_graph: bool = True, threshold: int = 8):
        self.layer = layer
        self.B = batch_size
        self.K = in_features
        self.N = out_features
        self.threshold = threshold
        self.use_graph = use_graph

        # Pre-allocate static memory (fixed addresses for CUDAGraph)
        self.static_X = torch.zeros(self.B, self.K, dtype=torch.float16, device="cuda")
        self.static_target = torch.zeros(self.B, self.N, dtype=torch.float16, device="cuda")
        self.static_pred = torch.zeros(self.B, self.N, dtype=torch.float16, device="cuda")
        self.static_dY = torch.zeros(self.B, self.N, dtype=torch.float16, device="cuda")
        self.static_dX = torch.zeros(self.B, self.K, dtype=torch.float16, device="cuda")

        # Snapshot initial state for reset
        self._W0 = layer.W_packed.clone()
        self._counter0 = layer.counter.clone()

        if use_graph:
            self._capture_graph()
        else:
            self.graph = None

    def _capture_graph(self):
        """Capture forward+backward_dx into a CUDAGraph."""
        # Warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                y = _forward_op(self.layer.W_packed, self.static_X, self.K)
                self.static_pred.copy_(y)
                self.static_dY.copy_(2.0 * (self.static_pred - self.static_target) / (self.B * self.N))
                dx = _backward_op(self.layer.W_packed, self.static_dY, self.K)
                self.static_dX.copy_(dx)
        torch.cuda.current_stream().wait_stream(s)

        # Capture
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            y = _forward_op(self.layer.W_packed, self.static_X, self.K)
            self.static_pred.copy_(y)
            self.static_dY.copy_(2.0 * (self.static_pred - self.static_target) / (self.B * self.N))
            dx = _backward_op(self.layer.W_packed, self.static_dY, self.K)
            self.static_dX.copy_(dx)

    def step(self, X: torch.Tensor, Y_target: torch.Tensor) -> float:
        """Execute one training step.

        Returns loss (before update).
        """
        if self.graph is not None:
            self.static_X.copy_(X)
            self.static_target.copy_(Y_target)
            self.graph.replay()
            Y_out = self.static_pred
            dY = self.static_dY
        else:
            # Manual path (no graph)
            Y_out = _forward_op(self.layer.W_packed, X.contiguous(), self.K)
            dY = 2.0 * (Y_out - Y_target) / (self.B * self.N)

        # Update runs outside graph using the computed dY, NOT Y_target
        _update_op(self.layer.W_packed, self.layer.counter,
                   X.contiguous(), dY, self.threshold)

        # Compute loss
        loss_val = (Y_out - Y_target).pow(2).mean().item()
        return loss_val

    def reset_weights(self):
        """Reset W and counter to initial state."""
        self.layer.W_packed.data.copy_(self._W0)
        self.layer.counter.data.copy_(self._counter0)
