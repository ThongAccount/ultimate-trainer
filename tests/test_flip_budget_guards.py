"""tests/test_flip_budget_guards.py — flip-budget guards from 07001be / 6192a47 / b96a6ba.

HOW TO RUN
----------
CPU box WITHOUT torch (torch-less check via py_compile, or standalone run):
    python -m pytest tests/test_flip_budget_guards.py -q
    # or standalone, no pytest needed:
    python tests/test_flip_budget_guards.py
  torch is not installed on such a box, so this file installs a minimal torch
  stub (torch, torch.library.custom_op, torch.Tensor/empty/zeros) into
  sys.modules BEFORE importing the modules under test, and loads
  kernels/packed_ternary/{custom_ops,pack_update,update_cublas}.py through
  synthetic parent packages so the heavy kernels/__init__.py never executes.
  Kernel dispatch is monkeypatched with recording fakes, so no CUDA compile
  and no GPU are needed — every dispatch test runs on plain CPU.

Colab T4 WITH real torch:
    python -m pytest tests/test_flip_budget_guards.py -q
  The stub is NOT installed (import torch succeeds). The exact same tests run
  against real torch on CPU tensors; dispatch is still monkeypatched, so no
  kernel is ever loaded and the tests pass on T4 without compiling anything.

WHICH TESTS USE STUB vs REAL TORCH
----------------------------------
All Python-dispatch tests (01-05) run under whichever torch is importable:
real torch on Colab, the in-file stub on a torch-less CPU box. Only the pure
source scans (04b, 06, 07) are torch-independent text checks and behave
identically everywhere.
"""

import contextlib
import importlib
import math
import os
import re
import sys
import types

# Default: repo root = parent of tests/. Override for out-of-tree verification:
#   ULTIMATE_TRAINER_REPO=/path/to/ultimate-trainer python tests/test_flip_budget_guards.py
REPO = os.environ.get(
    "ULTIMATE_TRAINER_REPO", os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# ── torch: real if importable, else a minimal stub ────────────────────────

try:
    import torch  # type: ignore
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False
    torch = None  # replaced by _install_torch_stub() below


class _FakeTensor:
    """Minimal torch.Tensor stand-in for torch-less CPU boxes."""

    def __init__(self, shape=(), dtype=None, device="cpu"):
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device

    def size(self, dim=None):
        return self.shape if dim is None else self.shape[dim]

    def numel(self):
        return math.prod(self.shape) if self.shape else 1

    def is_contiguous(self):
        return True

    def contiguous(self):
        return self

    def __repr__(self):
        return f"_FakeTensor(shape={self.shape}, dtype={self.dtype})"


class _StubCustomOp:
    """Stand-in for what torch.library.custom_op returns: callable, exposes
    .mutates_args and .register_fake like the real decorator."""

    def __init__(self, name, fn, mutates_args):
        self.name = name
        self._fn = fn
        self.mutates_args = tuple(mutates_args)
        self.__wrapped__ = fn
        self._fake = None

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)

    def register_fake(self, fn):
        self._fake = fn
        return fn


def _install_torch_stub():
    mod = types.ModuleType("torch")

    class Tensor(_FakeTensor):
        pass

    mod.Tensor = Tensor
    mod.int16 = "int16"
    mod.int32 = "int32"
    mod.float16 = "float16"
    mod.float32 = "float32"

    def empty(shape, dtype=None, device=None, **kw):
        return Tensor(shape, dtype=dtype, device=device or "cpu")

    def zeros(shape, dtype=None, device=None, **kw):
        return Tensor(shape, dtype=dtype, device=device or "cpu")

    mod.empty = empty
    mod.zeros = zeros

    library = types.ModuleType("torch.library")

    def custom_op(name, mutates_args=(), **kw):
        def deco(fn):
            return _StubCustomOp(name, fn, mutates_args)
        return deco

    library.custom_op = custom_op
    mod.library = library
    sys.modules["torch"] = mod
    sys.modules["torch.library"] = library
    return mod


if not HAS_TORCH:
    torch = _install_torch_stub()


# ── Load the modules under test without executing kernels/__init__.py ─────
# (kernels/__init__.py pulls in elementwise/ternary/fused_ternary extensions;
#  synthetic parent packages let us import just the packed_ternary modules.)

if "kernels" not in sys.modules:
    _k = types.ModuleType("kernels")
    _k.__path__ = [os.path.join(REPO, "kernels")]
    sys.modules["kernels"] = _k
if "kernels.packed_ternary" not in sys.modules:
    _p = types.ModuleType("kernels.packed_ternary")
    _p.__path__ = [os.path.join(REPO, "kernels", "packed_ternary")]
    sys.modules["kernels.packed_ternary"] = _p
    sys.modules["kernels"].packed_ternary = _p

custom_ops = importlib.import_module("kernels.packed_ternary.custom_ops")
pack_update = importlib.import_module("kernels.packed_ternary.pack_update")
update_cublas = importlib.import_module("kernels.packed_ternary.update_cublas")


# ── helpers ───────────────────────────────────────────────────────────────

def _t(shape, dtype="float16"):
    """Tensor factory: real torch on Colab, _FakeTensor on the stub path."""
    if HAS_TORCH:
        return torch.zeros(shape, dtype=getattr(torch, dtype))
    return _FakeTensor(shape, dtype=dtype)


def _shapes():
    W = _t((32, 8), "int32")        # [N=32, stride_words=8] packed weights
    counter = _t((32, 16), "int16")  # [N, K]
    flip_count = _t((32, 16), "int16")
    X = _t((16, 16), "float16")     # [B=16, K=16]
    dY = _t((16, 32), "float16")    # [B=16, N=32]
    return W, counter, flip_count, X, dY


@contextlib.contextmanager
def _patched(obj, **attrs):
    old = {k: getattr(obj, k) for k in attrs}
    for k, v in attrs.items():
        setattr(obj, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(obj, k, v)


@contextlib.contextmanager
def _env(**kw):
    old = {k: os.environ.get(k) for k in kw}
    for k, v in kw.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class _Recorder:
    def __init__(self, ret=None):
        self.calls = []
        self.ret = ret

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.ret


def assert_raises(exc, match, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc as e:
        if match is not None:
            assert re.search(match, str(e)), f"exception {e!r} does not match /{match}/"
        return e
    raise AssertionError(f"expected {exc.__name__} matching /{match}/, got no exception")


# ── 1. flip_budget>0 + flip_count=None raises BEFORE load and cublas ──────

def test_01_none_flip_count_with_budget_raises_before_load_and_cublas():
    W, counter, fc, X, dY = _shapes()

    def boom_load():
        raise AssertionError("_ensure_loaded must not run before the flip guards")

    def boom_cublas(*a, **k):
        raise AssertionError("cublas early-return must not run before the flip guards")

    # custom backend: guard fires before _ensure_loaded()
    with _patched(custom_ops, _ensure_loaded=boom_load), _env(PT_UPDATE_BACKEND=None):
        assert_raises(RuntimeError, "flip_count", custom_ops.update_tc_v2,
                      W, counter, X, dY, 64, dead_zone=1e-3,
                      flip_count=None, flip_budget=5)
    # cublas backend armed: guard must still fire before the early-return
    with _patched(custom_ops, _ensure_loaded=boom_load), \
         _patched(update_cublas, update_cublas=boom_cublas), \
         _env(PT_UPDATE_BACKEND="cublas"):
        assert_raises(RuntimeError, "flip_count", custom_ops.update_tc_v2,
                      W, counter, X, dY, 64, dead_zone=1e-3,
                      flip_count=None, flip_budget=5)


# ── 2. flip_budget outside [1,32767] raises, before the cublas early-return ─

def test_02_budget_range_guard_before_cublas_early_return():
    W, counter, fc, X, dY = _shapes()

    # 32768 > 32767 -> raise, with PT_UPDATE_BACKEND=cublas armed (guard is
    # at custom_ops.py:125-131, the cublas branch at :132).
    rec = _Recorder()
    with _patched(custom_ops, _ensure_loaded=lambda: None), \
         _patched(pack_update, _select_update_fn=lambda B, N, K: (rec, "v2_64")), \
         _env(PT_UPDATE_BACKEND="cublas"):
        assert_raises(RuntimeError, r"flip_budget must be in \[1, 32767\]",
                      custom_ops.update_tc_v2, W, counter, X, dY, 64,
                      dead_zone=1e-3, flip_count=fc, flip_budget=32768)
        assert rec.calls == [], "guard must raise before any dispatch"

    # negative budget -> raise on the custom path too (symmetric with
    # update_cublas, which rejects < 0)
    with _patched(custom_ops, _ensure_loaded=lambda: None), \
         _env(PT_UPDATE_BACKEND=None):
        assert_raises(RuntimeError, r"flip_budget must be in \[0, 32767\]",
                      custom_ops.update_tc_v2, W, counter, X, dY, 64,
                      dead_zone=1e-3, flip_count=fc, flip_budget=-1)

    # boundary values 1 and 32767 pass the guard and reach dispatch
    for budget in (1, 32767):
        rec = _Recorder()
        with _patched(custom_ops, _ensure_loaded=lambda: None), \
             _patched(pack_update, _select_update_fn=lambda B, N, K: (rec, "v2_64")), \
             _env(PT_UPDATE_BACKEND=None):
            custom_ops.update_tc_v2(W, counter, X, dY, 64, dead_zone=1e-3,
                                    flip_count=fc, flip_budget=budget)
            assert len(rec.calls) == 1, f"budget={budget} should pass the guard"
            assert rec.calls[0][0][7] == budget

    # wrong-shaped flip_count at budget>0 -> raise (shape guard)
    bad_fc = _t((32, 8), "int16")
    with _patched(custom_ops, _ensure_loaded=lambda: None), \
         _env(PT_UPDATE_BACKEND=None):
        assert_raises(RuntimeError, "must equal counter", custom_ops.update_tc_v2,
                      W, counter, X, dY, 64, dead_zone=1e-3,
                      flip_count=bad_fc, flip_budget=5)


# ── 3. arity dispatch: v2_* = 8 args (dummy flip_count at budget 0), v3/v1 = 5 ─

def test_03_arity_dispatch_custom_ops():
    W, counter, fc, X, dY = _shapes()
    for name in ("v2_64", "v2_32"):
        # flip_budget=0, flip_count=None -> 8-arg call with 1-elem dummy fc
        rec = _Recorder()
        with _patched(custom_ops, _ensure_loaded=lambda: None), \
             _patched(pack_update, _select_update_fn=lambda B, N, K: (rec, name)), \
             _env(PT_UPDATE_BACKEND=None):
            custom_ops.update_tc_v2(W, counter, X, dY, 64, dead_zone=1e-3,
                                    flip_count=None, flip_budget=0)
        args, _ = rec.calls[0]
        assert len(args) == 8, f"{name}: expected 8-arg call, got {len(args)}: {args}"
        assert args[0] is W and args[1] is counter
        assert args[2] is not None and args[2].numel() == 1, \
            f"{name}: flip_count=None at budget 0 must become a 1-elem dummy"
        assert args[5] == 64 and args[6] == 1e-3 and args[7] == 0
        # real flip_count + budget>0 -> passed through, still 8 args
        rec = _Recorder()
        with _patched(custom_ops, _ensure_loaded=lambda: None), \
             _patched(pack_update, _select_update_fn=lambda B, N, K: (rec, name)), \
             _env(PT_UPDATE_BACKEND=None):
            custom_ops.update_tc_v2(W, counter, X, dY, 64, dead_zone=1e-3,
                                    flip_count=fc, flip_budget=100)
        args, _ = rec.calls[0]
        assert len(args) == 8 and args[2] is fc and args[7] == 100
    for name in ("v3_32", "v1_32"):
        rec = _Recorder()
        with _patched(custom_ops, _ensure_loaded=lambda: None), \
             _patched(pack_update, _select_update_fn=lambda B, N, K: (rec, name)), \
             _env(PT_UPDATE_BACKEND=None):
            custom_ops.update_tc_v2(W, counter, X, dY, 64, dead_zone=1e-3,
                                    flip_count=None, flip_budget=0)
        args, _ = rec.calls[0]
        assert len(args) == 5, f"{name}: expected native 5-arg call, got {len(args)}"
        assert args[4] == 64
        # budget>0 on a non-v2 dispatch -> loud raise (custom_ops.py:154-156)
        with _patched(custom_ops, _ensure_loaded=lambda: None), \
             _patched(pack_update, _select_update_fn=lambda B, N, K: (rec, name)), \
             _env(PT_UPDATE_BACKEND=None):
            assert_raises(RuntimeError, r"only supported on v2_64/v2_32",
                          custom_ops.update_tc_v2, W, counter, X, dY, 64,
                          dead_zone=1e-3, flip_count=fc, flip_budget=7)
    # scalar fallback: budget>0 -> loud raise; budget=0 -> native 5-arg _up_fn
    rec_up = _Recorder()
    with _patched(custom_ops, _ensure_loaded=lambda: None), \
         _patched(pack_update, _select_update_fn=lambda B, N, K: (None, None),
                  _load_if_needed=lambda: None, _HAS_UP=True, _up_fn=rec_up), \
         _env(PT_UPDATE_BACKEND=None):
        assert_raises(RuntimeError, "scalar fallback", custom_ops.update_tc_v2,
                      W, counter, X, dY, 64, dead_zone=1e-3,
                      flip_count=fc, flip_budget=3)
        custom_ops.update_tc_v2(W, counter, X, dY, 64, dead_zone=1e-3,
                                flip_count=None, flip_budget=0)
    assert len(rec_up.calls) == 1 and len(rec_up.calls[0][0]) == 5


def test_03b_arity_dispatch_pack_update_update():
    W, counter, fc, X, dY = _shapes()
    for name in ("v2_64", "v2_32"):
        rec = _Recorder()
        with _patched(pack_update, _select_update_fn=lambda B, N, K: (rec, name)):
            pack_update.update(W, counter, X, dY, 64, flip_count=None, flip_budget=0)
        args, _ = rec.calls[0]
        assert len(args) == 8, f"update() {name}: expected 8 args, got {len(args)}"
        assert args[2] is not None and args[2].numel() == 1, "dummy flip_count"
        assert args[5] == 64 and args[6] == 0.0 and args[7] == 0, \
            "legacy path pins dead_zone=0.0 (pack_update.py:693-703)"
        rec = _Recorder()
        with _patched(pack_update, _select_update_fn=lambda B, N, K: (rec, name)):
            pack_update.update(W, counter, X, dY, 64, flip_count=fc, flip_budget=50)
        args, _ = rec.calls[0]
        assert len(args) == 8 and args[2] is fc and args[7] == 50
        # budget>0 + flip_count=None on v2 -> raise (pack_update.py:696-698)
        with _patched(pack_update, _select_update_fn=lambda B, N, K: (rec, name)):
            assert_raises(RuntimeError, "flip_count", pack_update.update,
                          W, counter, X, dY, 64, flip_count=None, flip_budget=9)
        # negative budget -> raise (symmetric with update_cublas)
        with _patched(pack_update, _select_update_fn=lambda B, N, K: (rec, name)):
            assert_raises(RuntimeError, r"flip_budget must be in \[0, 32767\]",
                          pack_update.update, W, counter, X, dY, 64,
                          flip_count=fc, flip_budget=-2)
    for name in ("v3_32", "v1_32"):
        rec = _Recorder()
        with _patched(pack_update, _select_update_fn=lambda B, N, K: (rec, name)):
            pack_update.update(W, counter, X, dY, 64, flip_count=None, flip_budget=0)
        args, _ = rec.calls[0]
        assert len(args) == 5, f"update() {name}: expected native 5 args, got {len(args)}"
        with _patched(pack_update, _select_update_fn=lambda B, N, K: (rec, name)):
            assert_raises(RuntimeError, r"only supported on v2_64/v2_32",
                          pack_update.update, W, counter, X, dY, 64,
                          flip_count=fc, flip_budget=9)
    # scalar fallback (pack_update.py:711-718)
    rec_up = _Recorder()
    with _patched(pack_update, _select_update_fn=lambda B, N, K: (None, None),
                  _load_if_needed=lambda: None, _HAS_UP=True, _up_fn=rec_up):
        assert_raises(RuntimeError, "scalar fallback", pack_update.update,
                      W, counter, X, dY, 64, flip_count=fc, flip_budget=9)
        pack_update.update(W, counter, X, dY, 64, flip_count=None, flip_budget=0)
    assert len(rec_up.calls) == 1 and len(rec_up.calls[0][0]) == 5


def test_03c_arity_dispatch_pack_update_backward_update():
    W, counter, fc, X, dY = _shapes()
    dX_ret = _t((16, 16))
    dx_rec = _Recorder(ret=dX_ret)
    # hermetic dX path: force the scalar _dx_fn recorder regardless of CUDA
    dx_patch = dict(_load_dx_tc_32=lambda: None, _load_dx_tc=lambda: None,
                    _load_if_needed=lambda: None,
                    _HAS_DX_TC_32=False, _HAS_DX_TC=False, _dx_fn=dx_rec)
    for name in ("v2_64", "v2_32", "v3_32", "v1_32"):
        rec = _Recorder()
        with _patched(pack_update, _select_update_fn=lambda B, N, K: (rec, name), **dx_patch):
            dX = pack_update.backward_update(W, counter, dY, X, 16, 64,
                                             flip_count=None, flip_budget=0)
        assert dX is dX_ret
        args, _ = rec.calls[0]
        if name in ("v2_64", "v2_32"):
            assert len(args) == 8, f"backward_update() {name}: expected 8 args, got {len(args)}"
            assert args[2] is not None and args[2].numel() == 1 and args[7] == 0
        else:
            assert len(args) == 5, f"backward_update() {name}: expected 5 args, got {len(args)}"
    # budget>0 + None on v2 -> raise (pack_update.py:763-765); budget>0 on v3 -> raise
    rec = _Recorder()
    with _patched(pack_update, _select_update_fn=lambda B, N, K: (rec, "v2_64"), **dx_patch):
        assert_raises(RuntimeError, "flip_count", pack_update.backward_update,
                      W, counter, dY, X, 16, 64, flip_count=None, flip_budget=9)
    rec = _Recorder()
    with _patched(pack_update, _select_update_fn=lambda B, N, K: (rec, "v3_32"), **dx_patch):
        assert_raises(RuntimeError, r"only supported on v2_64/v2_32",
                      pack_update.backward_update,
                      W, counter, dY, X, 16, 64, flip_count=fc, flip_budget=9)
    # negative budget -> raise in backward_update too
    rec = _Recorder()
    with _patched(pack_update, _select_update_fn=lambda B, N, K: (rec, "v2_64"), **dx_patch):
        assert_raises(RuntimeError, r"flip_budget must be in \[0, 32767\]",
                      pack_update.backward_update,
                      W, counter, dY, X, 16, 64, flip_count=fc, flip_budget=-3)


# ── 4. mutates_args contains "flip_count" on the custom_op ─────────────────

def test_04_mutates_args_on_op_object():
    op = custom_ops.update_tc_v2
    found = None
    for attr in ("mutates_args", "_mutates_args"):
        v = getattr(op, attr, None)
        if v:
            found = tuple(v)
            break
    if found is None:
        schema = getattr(op, "_schema", None)
        if schema is not None and hasattr(schema, "mutates_args"):
            try:
                found = tuple(schema.mutates_args)
            except Exception:
                found = None
    if found is not None:
        # stub always exposes it; real torch CustomOp exposes it on most versions
        assert "flip_count" in found and "W" in found and "counter" in found, found
    # else: not exposed on this torch version — the source test below pins it


def test_04b_mutates_args_declared_in_source():
    src = open(os.path.join(REPO, "kernels/packed_ternary/custom_ops.py"),
               encoding="utf-8").read()
    m = re.search(
        r'custom_op\(\s*"packed_ternary::update_tc_v2"\s*,\s*mutates_args\s*=\s*\(([^)]*)\)',
        src)
    assert m, "packed_ternary::update_tc_v2 custom_op declaration not found"
    names = [n.strip().strip('"\'') for n in m.group(1).split(",") if n.strip()]
    assert "flip_count" in names, f"mutates_args must contain flip_count: {names}"


# ── 5. update_cublas: None+budget raise (07001be), range guard, dummy at 0 ──

def test_05_update_cublas_guards():
    W, counter, fc, X, dY = _shapes()
    gemm_rec = _Recorder()
    epi_rec = _Recorder()
    base = dict(_HAS_UPDATE_CUBLAS=True,  # _load_update_cublas() returns at once
                _dw_gemm_f32=gemm_rec, _update_epilogue=epi_rec)
    # None + budget>0 -> raise (update_cublas.py:130-134)
    with _patched(update_cublas, **base):
        assert_raises(RuntimeError, "flip_count", update_cublas.update_cublas,
                      W, counter, X, dY, 32, 1e-3, None, 9)
    # budget outside [0,32767] -> raise (update_cublas.py:128-129)
    with _patched(update_cublas, **base):
        assert_raises(RuntimeError, r"flip_budget must be in \[0, 32767\]",
                      update_cublas.update_cublas, W, counter, X, dY, 32, 1e-3, fc, 32768)
        assert_raises(RuntimeError, r"flip_budget must be in \[0, 32767\]",
                      update_cublas.update_cublas, W, counter, X, dY, 32, 1e-3, fc, -1)
    # budget=0 + None -> no raise; 1-elem int16 dummy; epilogue gets budget 0
    with _patched(update_cublas, **base):
        dW = update_cublas.update_cublas(W, counter, X, dY, 32, 1e-3, None, 0)
    assert len(gemm_rec.calls) == 1, "dW GEMM must run"
    assert len(epi_rec.calls) == 1
    args, _ = epi_rec.calls[0]
    assert len(args) == 7, f"epilogue wrapper arity: {len(args)}"
    assert args[2] is not None and args[2].numel() == 1, \
        "flip_count=None at budget 0 must become a 1-elem dummy (update_cublas.py:140-141)"
    assert args[4] == 32 and args[5] == 1e-3 and args[6] == 0
    assert dW is not None


# ── 6. .cu kernels: every flip_count deref guarded by flip_budget > 0 ──────

CU_FILES = [
    "kernels/packed_ternary/gemm_update_tc_v2.cu",
    "kernels/packed_ternary/gemm_update_tc_v2_32.cu",
    "kernels/packed_ternary/gemm_update_cublas.cu",
]


def test_06_cu_kernels_guard_every_flip_count_deref():
    """flip_budget=0 must be behavior-identical: the kernels may only touch
    flip_count under `flip_budget > 0`, so a 1-element dummy pointer is safe."""
    for rel in CU_FILES:
        path = os.path.join(REPO, rel)
        with open(path, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
        derefs = [(i + 1, ln) for i, ln in enumerate(lines) if "flip_count[" in ln]
        assert derefs, f"{rel}: no flip_count derefs found — scan would be vacuous"
        bad = [(n, ln) for n, ln in derefs if "flip_budget > 0" not in ln]
        assert not bad, (
            f"{rel}: UNGUARDED flip_count derefs (flip_budget=0 would OOB): "
            + "; ".join(f"{n}: {ln.strip()}" for n, ln in bad)
        )


# ── 7. b96a6ba: CUDAGraph path threads flip_count/flip_budget by keyword ───

def test_07_train_step_graph_v2_threads_flip_args():
    src = open(os.path.join(REPO, "train_step_graph_v2.py"), encoding="utf-8").read()
    assert re.search(
        r"flip_count\s*=\s*self\.layer\.flip_count\s*,\s*flip_budget\s*=\s*flip_budget", src
    ), ("train_step_graph_v2.step must pass flip_count/flip_budget by keyword on the "
        "budgeted path (b96a6ba, train_step_graph_v2.py:98-105)")


# ── 8. 557fbd7: packed_linear backward fallback threads flip args too ──────

def test_08_packed_linear_backward_fallback_threads_flip_args():
    src = open(os.path.join(REPO, "kernels/packed_ternary/packed_linear.py"),
               encoding="utf-8").read()
    assert re.search(
        r"backward_update\(W_packed, counter, dY, X, ctx\.in_features, threshold,\s*"
        r"flip_count=flip_count, flip_budget=int\(flip_budget\)\)", src
    ), ("packed_linear.py backward fallback must thread flip_count/flip_budget into "
        "backward_update (no silent unbudgeted run)")


# ── standalone runner (no pytest needed) ──────────────────────────────────

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print("PASS", fn.__name__)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print("FAIL", fn.__name__, "-", repr(e))
    print(f"{len(fns) - failed} passed, {failed} failed  (HAS_TORCH={HAS_TORCH}, "
          f"stub={'no' if HAS_TORCH else 'yes'})")
    sys.exit(1 if failed else 0)
