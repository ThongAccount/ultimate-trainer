"""Run a child process and forward its output line by line.

Why this exists
---------------
`colab run` / `colab exec` execute the script inside a JUPYTER KERNEL. The
CLI captures what the kernel PRINTS, but a child process that merely
INHERITS stdout writes straight to the kernel server's file descriptor 1 —
which the CLI never sees. The output is silently lost.

That cost one full T4 allocation on 2026-10-26: the official 300-step A/B
and the 1,000-step flip-rate gate both ran to completion (exit codes 0 and
1) and their numbers went nowhere.

So: every child must be run with an explicit PIPE, and the parent must
re-print what it reads.  `readline()` is used instead of iterating the file
object, because iteration uses a read-ahead buffer that delays lines.
"""

from __future__ import annotations

import subprocess
import sys


def run_streaming(cmd, cwd=None, env=None, echo=True):
    """Run `cmd`, forwarding stdout+stderr line by line.

    Returns (returncode, lines) where `lines` is the full captured output.
    """
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines = []
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        line = line.rstrip("\n")
        lines.append(line)
        if echo:
            print(line, flush=True)
    proc.wait()
    return proc.returncode, lines
