"""BitLinear 2B4T spec with HF Kernels compat + QAT gamma caching."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from kernels import use_kernel_forward_from_hub
except ImportError:

    def use_kernel_forward_from_hub(name):
        def _decorator(cls):
            return cls

        return _decorator


# Fused Triton ternary matmul (GPU) with eager CPU fallback
try:
    from kernels.ternary_matmul import fused_bitlinear_forward, compute_gamma

    _HAS_FUSED_KERNEL = True
except ImportError:
    _HAS_FUSED_KERNEL = False


def absmax_quantize_activation(act, bits=8):
    q_max = float(2 ** (bits - 1) - 1)
    abs_max = act.abs().max(dim=-1, keepdim=True).values
    scale = (abs_max + 1e-8) / (q_max + 1e-5)
    act_quant = torch.clamp(torch.round(act / scale), -q_max, q_max)
    act_dequant = act_quant * scale
    return act + (act_dequant - act).detach()


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).sqrt()
        return x / (rms + self.eps) * self.weight


@use_kernel_forward_from_hub("BitLinear")
class BitLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=False,
        quantize_activations=True,
        activation_bits=8,
        quant_update_freq=10,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quantize_activations = quantize_activations
        self.activation_bits = activation_bits
        self.quant_update_freq = quant_update_freq
        self._quant_step = 0
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("_gamma", torch.ones(1))
        self.register_buffer("_w_ternary", torch.empty(out_features, in_features), persistent=False)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        self._w_ternary.copy_(self.weight.detach())

    def _refresh_ternary_weights(self):
        """Recompute cached gamma and ternary weights from current self.weight."""
        gamma = compute_gamma(self.weight)
        if gamma.ndim == 0:
            gamma = gamma.reshape(1)
        self._gamma = gamma
        w_q = torch.clamp(torch.round(self.weight / self._gamma), -1.0, 1.0)
        self._w_ternary = self.weight + (w_q - self.weight).detach()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys,
                              error_msgs):
        """After loading weights, recompute ternary cache from restored self.weight."""
        # _w_ternary is non-persistent; strip from both directions.
        if f"{prefix}_w_ternary" in state_dict:
            del state_dict[f"{prefix}_w_ternary"]
        elif f"{prefix}_w_ternary" in missing_keys:
            missing_keys.remove(f"{prefix}_w_ternary")
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata,
            strict, missing_keys, unexpected_keys,
            error_msgs,
        )
        self._refresh_ternary_weights()

    def eval(self):
        super().eval()
        self._refresh_ternary_weights()
        return self

    def forward(self, x):
        if self.quantize_activations and self.training:
            x = absmax_quantize_activation(x, bits=self.activation_bits)
        if self.training:
            self._quant_step += 1
            stale = self._quant_step % self.quant_update_freq == 1
            if stale:
                self._refresh_ternary_weights()

        if not self.training and self.quantize_activations and x.is_cuda and _HAS_FUSED_KERNEL:
            return fused_bitlinear_forward(x, self.weight, self._gamma, self.bias)
        return F.linear(x, self._w_ternary, self.bias)

    def extra_repr(self):
        return f"{self.in_features}x{self.out_features}, ternary"
