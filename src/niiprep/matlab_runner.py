import os
import subprocess


def _scrubbed_env(cores: int = None) -> dict:
    """Return a copy of the environment safe for launching MATLAB R2025a.

    Strips any user-local GCC lib paths that ship an older libstdc++ than
    MATLAB requires (needs GLIBCXX_3.4.29+, CXXABI_1.3.13+). These commonly
    leak in via LD_LIBRARY_PATH and/or an LD_PRELOAD of libstdc++.so.6
    (e.g. ~/opt/gcc-*/lib64) set in the user's shell rc, and cause MATLAB to
    fail to start with "version `GLIBCXX_3.4.29' not found" errors.
    """
    env = os.environ.copy()

    def _scrub(value: str) -> str:
        return ":".join(p for p in value.split(":") if "opt/gcc" not in p and p)

    env["LD_LIBRARY_PATH"] = _scrub(env.get("LD_LIBRARY_PATH", ""))
    env["LD_PRELOAD"] = _scrub(env.get("LD_PRELOAD", ""))

    # Cap OpenMP-threaded mex code (e.g. BM4D) at the requested core count.
    if cores is not None:
        env["OMP_NUM_THREADS"] = str(cores)

    return env


def run_matlab(matlab_cmd: str, cores: int = None) -> None:
    """Run a MATLAB command in batch mode with a sanitized environment."""
    subprocess.run(
        ["matlab", "-nodisplay", "-nosplash", "-r", matlab_cmd],
        check=True,
        env=_scrubbed_env(cores),
    )
