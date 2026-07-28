#!/usr/bin/env python3
"""SciPy single-core oracle for the Cephes-rational special kernels (erf, j0)
and the bessel-zero family. Times scipy.special on the SAME 65536-element grid
the fsci `special_array_65536` Criterion bench uses, plus jnjnp_zeros.

Usage: python3 docs/perf_oracle_special_cephes.py [--reps N] [--warmups W]
"""
import argparse
import hashlib
import pathlib
import statistics
import sys
import time

import numpy as np
import scipy
from scipy import special


def bench(fn, reps, warmups):
    for _ in range(warmups):
        fn()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return statistics.median(samples)


def _read_exact(stream, size):
    payload = bytearray()
    while len(payload) < size:
        chunk = stream.read(size - len(payload))
        if not chunk:
            raise EOFError(f"expected {size} bytes, received {len(payload)}")
        payload.extend(chunk)
    return bytes(payload)


def _write_line(text):
    sys.stdout.buffer.write(text.encode("utf-8") + b"\n")
    sys.stdout.buffer.flush()


def kv_live():
    """Persistent, exact-fixture SciPy arm for the half-integer kv harness."""
    scipy_root = pathlib.Path(scipy.__file__).resolve().parent
    special_file = pathlib.Path(special.__file__).resolve()
    fsci_loaded = any(name == "fsci" or name.startswith("fsci_") for name in sys.modules)
    genuine = (
        special.kv is scipy.special.kv
        and isinstance(special.kv, np.ufunc)
        and scipy_root in special_file.parents
        and not fsci_loaded
    )
    _write_line(
        f"READY scipy={scipy.__version__} numpy={np.__version__} "
        f"kv_name={special.kv.__name__} kv_type={type(special.kv).__name__} "
        f"special_file={special_file} fsci_loaded={fsci_loaded} genuine={genuine}"
    )

    order = None
    points = None
    result = None
    while True:
        command_bytes = sys.stdin.buffer.readline()
        if not command_bytes:
            return
        command = command_bytes.decode("utf-8").strip()
        fields = command.split()
        if not fields:
            continue
        if fields[0] == "PREP" and len(fields) == 3:
            order = float(fields[1])
            points = int(fields[2])
            payload = _read_exact(sys.stdin.buffer, points * 8)
            values = np.frombuffer(payload, dtype="<f8").copy()
            result = special.kv(order, values)
            _write_line(
                f"CASE order={order:.17g} points={points} "
                f"sorted={bool(np.all(values[1:] > values[:-1]))} "
                f"finite={bool(np.all(np.isfinite(values)))} "
                f"positive={bool(np.all(values > 0.0))} "
                f"input_sha256={hashlib.sha256(payload).hexdigest()}"
            )
        elif fields[0] == "PARITY" and len(fields) == 1:
            if result is None or points is None:
                _write_line("ERROR fixture-not-prepared")
                continue
            result = special.kv(order, values)
            payload = np.asarray(result, dtype="<f8").tobytes(order="C")
            _write_line(f"RESULT components={result.size}")
            sys.stdout.buffer.write(payload)
            sys.stdout.buffer.write(b"\n")
            sys.stdout.buffer.flush()
        elif fields[0] == "SOLVE" and len(fields) == 2:
            if result is None or points is None:
                _write_line("ERROR fixture-not-prepared")
                continue
            repetitions = int(fields[1])
            started = time.perf_counter()
            for _ in range(repetitions):
                result = special.kv(order, values)
            elapsed = time.perf_counter() - started
            checksum = float(result[0] + result[result.size // 2] + result[-1])
            _write_line(f"TIME {elapsed:.17g} {result.size} {checksum:.17g}")
        elif fields[0] == "QUIT":
            return
        else:
            _write_line(f"ERROR unsupported-command={command}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--warmups", type=int, default=10)
    args = ap.parse_args()

    # Mirror the Rust bench grid: 0.5 + i*0.0001 for i in 0..65536.
    xs = 0.5 + np.arange(65536, dtype=np.float64) * 0.0001

    print(f"scipy {special.__name__} oracle, n={xs.size}, reps={args.reps}")
    for name, fn in [
        ("erf", lambda: special.erf(xs)),
        ("j0", lambda: special.j0(xs)),
        ("gamma", lambda: special.gamma(xs)),
    ]:
        p50 = bench(fn, args.reps, args.warmups)
        print(f"  {name:8s} p50 = {p50*1e6:12.3f} us")

    # jnjnp_zeros equivalent: scipy.special.jnjnp_zeros(nt)
    for nt in (64, 128):
        p50 = bench(lambda: special.jnjnp_zeros(nt), max(20, args.reps // 4), 3)
        print(f"  jnjnp_zeros(nt={nt}) p50 = {p50*1e6:12.3f} us")


if __name__ == "__main__":
    if "--kv-live" in sys.argv[1:]:
        kv_live()
    else:
        main()
