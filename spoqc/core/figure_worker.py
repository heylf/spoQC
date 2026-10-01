"""Initializer of the figure worker processes (spoqc.core.figures).

Kept apart from figures.py: unpickling the initializer imports only this module and
spoqc.core.threads, so threads.configure(1) runs before the worker imports numpy, polars or numba.
"""

import os
import threading
import time

from spoqc.core import threads

# How often a worker checks that the process that started it is still alive.
PARENT_POLL_SECONDS = 1.0


def _exit_when_orphaned(parent_pid):
    while os.getppid() == parent_pid:
        time.sleep(PARENT_POLL_SECONDS)
    os._exit(1)


def init(parent_pid):
    """Run single-threaded, and exit once `parent_pid` is gone (the worker is re-parented)."""
    threads.configure(1)
    threading.Thread(
        target=_exit_when_orphaned, args=(parent_pid,), daemon=True
    ).start()
