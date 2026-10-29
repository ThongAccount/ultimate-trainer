"""CPU-only guard: pybind embedder decl/call arity must match the .cu launcher
definitions, for every launch_* symbol.

Why this exists: the 07001be-class bug (14-arg call vs 13-param decl vs
13-param .cu def) fails INSIDE load_inline at JIT-compile time, is swallowed
by the loader's try/except (flag stays False), and only surfaces on a GPU box
as a silent dispatch degradation. This test parses the sources as text — no
torch, no CUDA, no compile — so it runs anywhere, including CI on CPU.

Covers the live embedders (kernels/packed_ternary/*.py, build_kernels.py,
tests/probe_update_*.py) against every launcher definition in
kernels/**/*.cu and tests/*.cu. Also pins the PARAMETER ORDER of the v2
update launchers (flip_count immediately after counter) — 07001be was an
order bug that a bare count would not catch.

Runs under pytest and standalone (`python tests/test_decl_arity.py`).
"""

import os
import re
import sys

REPO = os.path.join(os.path.dirname(__file__), "..")

EMBEDDER_PYS = [
    "kernels/packed_ternary/pack_update.py",
    "kernels/packed_ternary/pack_forward.py",
    "kernels/packed_ternary/pack_encode.py",
    "kernels/packed_ternary/update_cublas.py",
    "build_kernels.py",
    "tests/probe_update_pipeline.py",
    "tests/probe_update_syncwarp2.py",
]

CU_GLOBS = [
    "kernels/packed_ternary/*.cu",
    "kernels/packed_ternary/*.cuh",
    "tests/*.cu",
]

V2_UPDATE_LAUNCHERS = {
    "launch_packed_ternary_update_tc_v2_64",
    "launch_packed_ternary_update_tc_v2",
}


def _read(path):
    with open(os.path.join(REPO, path), encoding="utf-8") as fh:
        return fh.read()


def _strip_comments(s):
    return re.sub(r"//[^\n]*", "", s)


def _count_top(s):
    """Count top-level comma-separated items in a param/arg list."""
    s = _strip_comments(s)
    if not s.strip():
        return 0
    depth = 0
    n = 1
    for ch in s:
        if ch in "(<":
            depth += 1
        elif ch in ")>":
            depth -= 1
        elif ch == "," and depth == 0:
            n += 1
    return n


def _param_names(s):
    """Rough param-name extraction: last identifier of each top-level item."""
    s = _strip_comments(s)
    items, depth, cur = [], 0, ""
    for ch in s:
        if ch in "(<":
            depth += 1
        elif ch in ")>":
            depth -= 1
        if ch == "," and depth == 0:
            items.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        items.append(cur)
    names = []
    for it in items:
        ids = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", it)
        names.append(ids[-1] if ids else "")
    return names


def _balanced(text, start):
    """Return the text between the '(' at start-1 and its matching ')'."""
    depth = 1
    j = start
    while depth:
        if text[j] == "(":
            depth += 1
        elif text[j] == ")":
            depth -= 1
        j += 1
    return text[start:j - 1]


def _decls(text):
    """{name: (nparams, param_names)} for `void name(` declarations."""
    out = {}
    for m in re.finditer(r"void\s+(launch_[A-Za-z0-9_]+)\s*\(", text):
        body = _balanced(text, m.end())
        out[m.group(1)] = (_count_top(body), _param_names(body))
    return out


def _calls(text, name, after):
    """[(offset, nargs)] for call sites of name after position `after`."""
    out = []
    for m in re.finditer(re.escape(name) + r"\s*\(", text[after:]):
        body = _balanced(text, after + m.end())
        out.append((after + m.start(), _count_top(body)))
    return out


def _cu_defs():
    """{name: [(file, nparams, param_names)]} for `extern \"C\" void name(` defs."""
    import glob
    out = {}
    for pattern in CU_GLOBS:
        for path in sorted(glob.glob(os.path.join(REPO, pattern))):
            text = _read(os.path.relpath(path, REPO))
            for m in re.finditer(
                r'extern\s+"C"\s+void\s+(launch_[A-Za-z0-9_]+)\s*\(', text
            ):
                body = _balanced(text, m.end())
                out.setdefault(m.group(1), []).append(
                    (os.path.relpath(path, REPO), _count_top(body), _param_names(body))
                )
    return out


def _embedder_decls_and_calls():
    """{name: {"decl": (file, n, names), "calls": [(file, nargs)]}}"""
    out = {}
    for rel in EMBEDDER_PYS:
        text = _read(rel)
        decls = _decls(text)
        for name, (n, names) in decls.items():
            entry = out.setdefault(name, {"decl": None, "calls": []})
            entry["decl"] = (rel, n, names)
            # calls: every occurrence of name( that is not the decl itself
            for m in re.finditer(re.escape(name) + r"\s*\(", text):
                if text[max(0, m.start() - 5):m.start()].strip().endswith("void"):
                    continue
                body = _balanced(text, m.end())
                entry["calls"].append((rel, _count_top(body)))
    return out


def test_py_decl_arity_matches_cu_def():
    defs = _cu_defs()
    embedders = _embedder_decls_and_calls()
    problems = []
    for name, entry in embedders.items():
        rel, n_decl, _ = entry["decl"]
        if name not in defs:
            problems.append(f"{name}: declared in {rel} but no .cu definition found")
            continue
        for cu_file, n_def, _ in defs[name]:
            if n_def != n_decl:
                problems.append(
                    f"{name}: decl {n_decl} params in {rel} != .cu def {n_def} in {cu_file}"
                )
    assert not problems, "\n".join(problems)


def test_call_arity_matches_decl():
    embedders = _embedder_decls_and_calls()
    problems = []
    for name, entry in embedders.items():
        rel, n_decl, _ = entry["decl"]
        for cfile, nargs in entry["calls"]:
            if nargs != n_decl:
                problems.append(
                    f"{name}: call with {nargs} args in {cfile} != decl {n_decl} params"
                )
    assert not problems, "\n".join(problems)


def test_v2_update_launchers_13_params_flip_count_after_counter():
    defs = _cu_defs()
    embedders = _embedder_decls_and_calls()
    for name in V2_UPDATE_LAUNCHERS:
        # every definition and every embedder decl: 13 params, and the
        # param named flip_count must sit immediately after counter
        all_sigs = [d for d in defs.get(name, [])]
        if name in embedders:
            all_sigs.append(("<embedder>", embedders[name]["decl"][1],
                             embedders[name]["decl"][2]))
        assert all_sigs, f"{name} not found"
        for src, n, names in all_sigs:
            assert n == 13, f"{name}: {n} params in {src}, expected 13"
            assert "flip_count" in names and "counter" in names, (src, names)
            assert names.index("flip_count") == names.index("counter") + 1, (
                f"{name}: flip_count must directly follow counter in {src}: {names}"
            )


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print("PASS", fn.__name__)
        except AssertionError as e:
            failed += 1
            print("FAIL", fn.__name__, "-", e)
    print(f"{len(fns) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
