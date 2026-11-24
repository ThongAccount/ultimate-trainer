"""Flip-count batching validation: parity + A/B on one T4.

Usage (on Colab):
    python scripts/colab_flip_count_validate.py --expect-commit COMMIT

Gates (fixed order):
  0. smoke   — tests/test_packed_ternary.py
  1. pytest  — colab_pytest.TEST_MODULES (includes update semantics + flip budget tests)
  2. parity  — bit-exact W trajectory vs saved reference from master
  3. ab      — same-VM whole-step A/B vs shipped baseline with flip_budget=5
"""
import argparse, os, subprocess, sys

CLONE = "/content/ultimate-trainer"
BRANCH = "feat/flip-count-batching"

def smoke():
    print("=== SMOKE ===", flush=True)
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_packed_ternary.py", "-x", "-q"],
                       cwd=CLONE, capture_output=True, text=True,
                       env={**os.environ, "UT_PREBUILT": "0"})
    print(r.stdout[-2000:] if r.stdout else "", flush=True)
    print(r.stderr[-1000:] if r.stderr else "", flush=True)
    return r.returncode

def pytest():
    print("=== PYTEST ===", flush=True)
    r = subprocess.run([sys.executable, "colab_pytest.py"],
                       cwd=CLONE, capture_output=True, text=True,
                       env={**os.environ, "UT_PREBUILT": "0"})
    print(r.stdout[-2000:] if r.stdout else "", flush=True)
    print(r.stderr[-1000:] if r.stderr else "", flush=True)
    return r.returncode

def parity():
    """Run the flip-budget-specific tests that verify W trajectory and flip_count semantics."""
    print("=== FLIP-COUNT PARITY (update semantics + flip budget guards) ===", flush=True)
    r = subprocess.run([sys.executable, "-m", "pytest",
                        "tests/test_update_semantics.py",
                        "tests/test_flip_budget_guards.py",
                        "tests/test_counter_clamp.py",
                        "tests/test_gemm_update.py",
                        "-x", "-q", "-v"],
                       cwd=CLONE, capture_output=True, text=True,
                       env={**os.environ, "UT_PREBUILT": "0"})
    print(r.stdout[-3000:] if r.stdout else "", flush=True)
    print(r.stderr[-1000:] if r.stderr else "", flush=True)
    return r.returncode

def ab():
    print("=== FLIP-COUNT A/B ===", flush=True)
    r = subprocess.run([sys.executable, "colab_perfgate.py", "--branch", BRANCH],
                       cwd=CLONE, capture_output=True, text=True,
                       env={**os.environ, "UT_PREBUILT": "0", "BRANCH": BRANCH,
                            "PT_FLIP_BUDGET": "5"})
    print(r.stdout[-2000:] if r.stdout else "", flush=True)
    print(r.stderr[-1000:] if r.stderr else "", flush=True)
    return r.returncode

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--expect-commit", required=True)
    p.add_argument("--gate", choices=["smoke", "pytest", "parity", "ab", "all"], default="all")
    args = p.parse_args()

    # Verify commit
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=CLONE, capture_output=True, text=True)
    actual = r.stdout.strip()
    if actual != args.expect_commit:
        print(f"COMMIT MISMATCH: expected {args.expect_commit}, got {actual}", flush=True)
        sys.exit(2)

    gates = {"smoke": smoke, "pytest": pytest, "parity": parity, "ab": ab}
    if args.gate == "all":
        for name in ["smoke", "pytest", "parity", "ab"]:
            rc = gates[name]()
            if rc != 0:
                print(f"GATE {name} FAILED (rc={rc})", flush=True)
                sys.exit(rc)
    else:
        rc = gates[args.gate]()
        sys.exit(rc)
