"""Native locks and process isolation with identical failure semantics."""
from contextlib import contextmanager
import os
from pathlib import Path
import subprocess

from . import PortabilityError


def process_options(detached=False, platform=None):
    platform = platform or os.name
    if platform == "nt":
        # Values are stable Win32 flags; available for cross-platform tests too.
        flags = 0x08000000  # CREATE_NO_WINDOW
        if detached:
            flags |= 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        return {"creationflags": flags, "close_fds": True}
    if platform == "posix":
        return {"start_new_session": detached, "close_fds": True}
    raise PortabilityError("unsupported process platform")


class ProcessLock:
    """Nonblocking lifetime lock. Never unlink the lock inode after release."""
    def __init__(self, path):
        self.path = Path(path)
        self.stream = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+b")
        try:
            if self.path.stat().st_size == 0:
                self.stream.write(b"\0"); self.stream.flush()
            self.stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            elif os.name == "posix":
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:
                raise PortabilityError("unsupported locking platform")
        except BaseException as error:
            self.stream.close(); self.stream = None
            raise PortabilityError("exclusive process lock unavailable") from error
        return self

    def __exit__(self, *exc):
        try:
            if os.name == "nt":
                import msvcrt
                self.stream.seek(0)
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        finally:
            self.stream.close(); self.stream = None


@contextmanager
def prevent_sleep():
    if os.name != "nt":
        yield  # Container lifetime is managed by its host, independently of this API.
        return
    import ctypes
    call = ctypes.windll.kernel32.SetThreadExecutionState
    if not call(0x80000001):
        raise PortabilityError("Windows sleep inhibition failed")
    try:
        yield
    finally:
        if not call(0x80000000):
            raise PortabilityError("Windows sleep inhibition cleanup failed")


def run_logged(argv, stdout, stderr, *, cwd):
    """Internal worker primitive; caller must hold authorization and registry claim."""
    if not isinstance(argv, list) or not argv or any(not isinstance(x, str) for x in argv):
        raise PortabilityError("explicit argument vector required")
    with Path(stdout).open("xb") as out, Path(stderr).open("xb") as err:
        result = subprocess.run(argv, cwd=cwd, stdin=subprocess.DEVNULL,
                                stdout=out, stderr=err, shell=False, **process_options())
    return result.returncode
