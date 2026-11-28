"""Machine-wide GPU exclusivity lock to prevent cross-session/cross-project MLX OOM & contention."""
import fcntl
import os
import subprocess
import time
from contextlib import contextmanager

LOCK_PATH = "/tmp/mlx_gpu.lock"
LEGACY_GPU_PATTERNS = ("agent8b.py", "mech.py", "ext.py", "seam_check.py")


def legacy_gpu_pids():
    """Return PIDs of legacy MLX scripts currently running (excluding self)."""
    me = os.getpid()
    pids = []
    try:
        out = subprocess.check_output(["ps", "-eo", "pid,command"], text=True)
    except Exception:
        return pids
    for line in out.splitlines()[1:]:
        parts = line.strip().split(None, 1)
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if pid == me:
            continue
        cmd = parts[1]
        if "python" in cmd and any(pat in cmd for pat in LEGACY_GPU_PATTERNS):
            pids.append((pid, cmd[:120]))
    return pids


@contextmanager
def gpu_lock(label="v2_job", wait=True, poll_s=15.0, lock_path=LOCK_PATH):
    """Acquire exclusive OS-level lock + wait for any legacy MLX script to finish."""
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o666)
    waited = False
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                if not wait:
                    raise RuntimeError(f"GPU lock {lock_path} is held by another process")
                if not waited:
                    print(f"[gpu_lock] waiting for {lock_path}...", flush=True)
                    waited = True
                time.sleep(poll_s)
                continue

            # Also ensure no legacy un-locked MLX job is holding Metal memory
            legacy = legacy_gpu_pids()
            if legacy:
                fcntl.flock(fd, fcntl.LOCK_UN)
                if not wait:
                    raise RuntimeError(f"Legacy MLX GPU process active: {legacy}")
                if not waited:
                    print(f"[gpu_lock] waiting for legacy GPU job(s): {legacy}", flush=True)
                    waited = True
                time.sleep(poll_s)
                continue
            break

        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, f"pid={os.getpid()} label={label} start={time.strftime('%F %T')}\n".encode())
        os.fsync(fd)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
