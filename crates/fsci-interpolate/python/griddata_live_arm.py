"""Persistent SciPy arm for the griddata live harness.

Reads the fixture ONCE, then serves one timed `griddata` call per stdin line so the two arms can
be interleaved inside a single invocation rather than compared across processes.

Protocol
--------
  stdin   "run <method>\\n"     -> stdout "t <elapsed_seconds> <checksum>\\n"
          "solve <method>\\n"   -> stdout "v <n> <v0> <v1> ...\\n"   (values, for the parity gate)
          "quit\\n"             -> exit

Startup prints a READY line naming the SciPy it actually loaded, so a row can record which
incumbent answered rather than which one was installed.
"""
import struct
import sys
import time
import hashlib
import os

import numpy as np
import scipy
from scipy.interpolate import griddata
from scipy.interpolate import _interpnd


def load(path):
    with open(path, "rb") as fh:
        buf = fh.read()
    np_, nq = struct.unpack_from("<QQ", buf, 0)
    off = 16
    pts = np.frombuffer(buf, dtype="<f8", count=np_ * 2, offset=off).reshape(np_, 2)
    off += np_ * 2 * 8
    vals = np.frombuffer(buf, dtype="<f8", count=np_, offset=off)
    off += np_ * 8
    xi = np.frombuffer(buf, dtype="<f8", count=nq * 2, offset=off).reshape(nq, 2)
    return np.ascontiguousarray(pts), np.ascontiguousarray(vals), np.ascontiguousarray(xi)


def main():
    pts, vals, xi = load(sys.argv[1])
    with open(_interpnd.__file__, "rb") as fh:
        engine_sha256 = hashlib.sha256(fh.read()).hexdigest()
    print(f"READY scipy={scipy.__version__} numpy={np.__version__} "
          f"engine_file={_interpnd.__file__} scipy_engine_sha256={engine_sha256} "
          f"actual_observed_scipy_threads={len(os.listdir('/proc/self/task'))} "
          f"npoints={len(pts)} nqueries={len(xi)} "
          f"fsci_loaded={'fsci' in sys.modules}", flush=True)

    # Warm every method once so the timed calls do not pay import/JIT-style first-call costs.
    for m in ("linear", "nearest", "cubic"):
        griddata(pts, vals, xi, method=m)

    for line in sys.stdin:
        parts = line.split()
        if not parts or parts[0] == "quit":
            break
        cmd, method = parts[0], parts[1]
        if cmd == "run":
            t0 = time.perf_counter()
            out = griddata(pts, vals, xi, method=method)
            dt = time.perf_counter() - t0
            acc = float(np.nansum(out))
            print(f"t {dt:.12f} {acc:.17e}", flush=True)
        elif cmd == "solve":
            out = np.asarray(griddata(pts, vals, xi, method=method), dtype=float)
            body = " ".join("nan" if np.isnan(v) else f"{v:.17e}" for v in out)
            print(f"v {out.size} {body}", flush=True)
        else:
            print("err unknown", flush=True)


if __name__ == "__main__":
    main()
