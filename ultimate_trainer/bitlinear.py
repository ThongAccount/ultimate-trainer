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
        self._quant_step = 5000  # start past warmup (alpha=1.0); set to 0 before training loop to enable ramp
        self.activation_warmup_steps = 5000
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
        """Recompute cached gamma and ternary weights from current self.weight.

        Uses .copy_() / .fill_() so registered buffers remain tracked
        (avoids the RuntimeError from reassigning self._w_ternary = ...).
        """
        gamma = compute_gamma(self.weight)
        if gamma.ndim == 0:
            gamma = gamma.reshape(1)
        self._gamma.copy_(gamma)
        w_q = torch.clamp(torch.round(self.weight / self._gamma), -1.0, 1.0)
        w_ste = self.weight + (w_q - self.weight).detach()
        self._w_ternary.copy_(w_ste)

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
            # Gradual quantization warmup: linearly ramp from 0% to 100%
            # over activation_warmup_steps (default 5000).
            alpha = min(1.0, (self._quant_step + 1) / max(1, self.activation_warmup_steps))
            if alpha > 0:
                x_q = absmax_quantize_activation(x, bits=self.activation_bits)
                x = x + (x_q - x).detach() * alpha

        if self.training:
            self._quant_step += 1
            stale = self._quant_step % self.quant_update_freq == 1
            if stale:
                # Sync cached ternary weights for eval readiness
                self._refresh_ternary_weights()
            # Fresh STE: forward uses ternary {-1,0,+1}, backward identity through self.weight.
            # This preserves the autograd graph — unlike the cached buffer (eval only).
            w_q = torch.clamp(torch.round(self.weight / self._gamma), -1.0, 1.0)
            w_ste = self.weight + (w_q - self.weight).detach()
            return F.linear(x, w_ste, self.bias)

        # Eval: use cached ternary buffer (fast, single memory read)
        return F.linear(x, self._w_ternary, self.bias)

    def extra_repr(self):
        return f"{self.in_features}x{self.out_features}, ternary"
