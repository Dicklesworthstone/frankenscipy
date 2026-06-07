#!/usr/bin/env python3
"""Scipy baseline benchmarks for comparison with FrankenSciPy.

Run with: python bench/scipy_baseline.py --json > bench/scipy_baseline.json
"""

import argparse
import json
import time
import statistics
import numpy as np
from dataclasses import dataclass
from typing import Callable


@dataclass
class BenchResult:
    name: str
    mean_ns: float
    median_ns: float
    p95_ns: float
    p99_ns: float
    min_ns: float
    max_ns: float
    runs: int


def bench(name: str, func: Callable, warmup: int = 5, runs: int = 100) -> BenchResult:
    for _ in range(warmup):
        func()

    times = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        func()
        elapsed = time.perf_counter_ns() - start
        times.append(elapsed)

    times.sort()
    return BenchResult(
        name=name,
        mean_ns=statistics.mean(times),
        median_ns=statistics.median(times),
        p95_ns=times[int(len(times) * 0.95)],
        p99_ns=times[int(len(times) * 0.99)],
        min_ns=min(times),
        max_ns=max(times),
        runs=runs,
    )


def run_fft_benchmarks():
    from scipy import fft as scipy_fft

    results = []
    sizes = [1024, 4096, 16384, 65536]

    for n in sizes:
        x_complex = np.sin(2 * np.pi * np.arange(n) / n) + 0j
        x_real = np.sin(2 * np.pi * np.arange(n) / n)

        results.append(bench(f"fft/{n}", lambda x=x_complex: scipy_fft.fft(x)))
        results.append(bench(f"rfft/{n}", lambda x=x_real: scipy_fft.rfft(x)))
        results.append(bench(f"ifft/{n}", lambda x=x_complex: scipy_fft.ifft(x)))

    # 2D FFT
    for dim in [64, 128, 256]:
        x = np.random.randn(dim, dim) + 0j
        results.append(bench(f"fft2/{dim}x{dim}", lambda x=x: scipy_fft.fft2(x)))

    return results


def run_linalg_benchmarks():
    from scipy import linalg

    results = []
    sizes = [100, 500, 1000, 2000]

    for n in sizes:
        # Diagonally dominant matrix
        A = np.eye(n) * n * 2
        for i in range(n):
            for j in range(n):
                if i != j:
                    A[i, j] = 1.0 / (abs(i - j) + 1)
        b = np.arange(1, n + 1, dtype=float)

        results.append(bench(f"solve/{n}x{n}", lambda A=A.copy(), b=b.copy(): linalg.solve(A, b)))
        results.append(bench(f"inv/{n}x{n}", lambda A=A.copy(): linalg.inv(A)))
        results.append(bench(f"det/{n}x{n}", lambda A=A.copy(): linalg.det(A)))

    # Least squares (overdetermined)
    for n in [100, 500, 1000]:
        rows = n * 2
        A = np.random.randn(rows, n)
        b = np.arange(1, rows + 1, dtype=float)
        results.append(bench(f"lstsq/{rows}x{n}", lambda A=A.copy(), b=b.copy(): linalg.lstsq(A, b)))

    return results


def run_optimize_benchmarks():
    from scipy import optimize

    results = []

    def rosenbrock(x):
        s = 0.0
        for i in range(len(x) - 1):
            s += 100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1.0 - x[i]) ** 2
        return s

    for dim in [2, 5, 10]:
        x0 = np.zeros(dim)
        results.append(bench(
            f"bfgs/rosenbrock_{dim}d",
            lambda x0=x0.copy(): optimize.minimize(rosenbrock, x0, method='BFGS'),
            runs=50
        ))
        results.append(bench(
            f"cg/rosenbrock_{dim}d",
            lambda x0=x0.copy(): optimize.minimize(rosenbrock, x0, method='CG'),
            runs=50
        ))

    # Root finding
    def cubic(x):
        return x ** 3 - 2 * x - 5

    results.append(bench("brentq/cubic", lambda: optimize.brentq(cubic, 1, 3)))
    results.append(bench("bisect/cubic", lambda: optimize.bisect(cubic, 1, 3)))

    return results


def run_special_benchmarks():
    from scipy import special

    results = []

    # Scalar benchmarks
    gamma_inputs = [0.5, 1.0, 2.5, 5.0, 10.0, 50.0, 100.0]
    for x in gamma_inputs:
        results.append(bench(f"gamma/{x}", lambda x=x: special.gamma(x)))
        results.append(bench(f"gammaln/{x}", lambda x=x: special.gammaln(x)))

    erf_inputs = [-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0]
    for x in erf_inputs:
        results.append(bench(f"erf/{x}", lambda x=x: special.erf(x)))

    bessel_inputs = [0.1, 1.0, 5.0, 10.0, 20.0]
    for x in bessel_inputs:
        results.append(bench(f"j0/{x}", lambda x=x: special.j0(x)))
        results.append(bench(f"y0/{x}", lambda x=x: special.y0(x)))

    # Vectorized
    x_vec = np.linspace(0.1, 10.0, 1000)
    results.append(bench("gamma_vec/1000", lambda x=x_vec: special.gamma(x)))
    results.append(bench("erf_vec/1000", lambda x=x_vec: special.erf(x)))
    results.append(bench("j0_vec/1000", lambda x=x_vec: special.j0(x)))

    return results


def run_signal_benchmarks():
    from scipy import signal

    results = []

    # FIR filter design and application
    for n in [64, 256, 1024]:
        b = signal.firwin(65, 0.3)
        x = np.random.randn(n)
        results.append(bench(f"lfilter/{n}", lambda b=b, x=x.copy(): signal.lfilter(b, [1.0], x)))

    # IIR filter (butterworth)
    sos = signal.butter(4, 0.3, output='sos')
    for n in [64, 256, 1024]:
        x = np.random.randn(n)
        results.append(bench(f"sosfilt/{n}", lambda sos=sos, x=x.copy(): signal.sosfilt(sos, x)))

    # Convolution
    for n in [256, 1024, 4096]:
        x = np.random.randn(n)
        h = np.random.randn(64)
        results.append(bench(f"convolve/{n}", lambda x=x, h=h: signal.convolve(x, h)))

    return results


def run_integrate_benchmarks():
    from scipy import integrate

    results = []

    def f1(x):
        return np.exp(-x ** 2)

    def f2(x):
        return np.sin(x) / x if x != 0 else 1.0

    results.append(bench("quad/gaussian", lambda: integrate.quad(f1, -5, 5), runs=50))
    results.append(bench("quad/sinc", lambda: integrate.quad(f2, 0.001, 10), runs=50))

    # Fixed-sample integration
    for n in [100, 1000, 10000]:
        x = np.linspace(0, 2 * np.pi, n)
        y = np.sin(x)
        results.append(bench(f"trapezoid/{n}", lambda x=x, y=y: integrate.trapezoid(y, x)))
        results.append(bench(f"simpson/{n}", lambda x=x, y=y: integrate.simpson(y, x=x)))

    return results


def run_stats_benchmarks():
    from scipy import stats

    results = []

    # PDF/CDF evaluation
    x = np.linspace(-5, 5, 1000)
    results.append(bench("norm_pdf/1000", lambda x=x: stats.norm.pdf(x)))
    results.append(bench("norm_cdf/1000", lambda x=x: stats.norm.cdf(x)))
    results.append(bench("t_pdf/1000", lambda x=x: stats.t.pdf(x, df=5)))

    # Statistical tests
    for n in [100, 1000, 10000]:
        a = np.random.randn(n)
        b = np.random.randn(n) + 0.1
        results.append(bench(f"ttest_ind/{n}", lambda a=a, b=b: stats.ttest_ind(a, b)))
        results.append(bench(f"pearsonr/{n}", lambda a=a, b=b: stats.pearsonr(a, b)))

    return results


def run_sparse_benchmarks():
    from scipy import sparse

    results = []

    for n in [1000, 5000, 10000]:
        density = 0.01
        A = sparse.random(n, n, density=density, format='csr')
        x = np.random.randn(n)
        results.append(bench(f"csr_matvec/{n}", lambda A=A, x=x: A @ x))

    # Sparse solve
    for n in [1000, 5000]:
        A = sparse.diags([-1, 4, -1], [-1, 0, 1], shape=(n, n), format='csr')
        b = np.ones(n)
        results.append(bench(f"spsolve/{n}", lambda A=A, b=b: sparse.linalg.spsolve(A, b), runs=50))

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', action='store_true', help='Output JSON')
    parser.add_argument('--runs', type=int, default=100, help='Number of runs per benchmark')
    parser.add_argument('--filter', type=str, help='Only run benchmarks matching filter')
    args = parser.parse_args()

    all_results = []

    benchmark_groups = [
        ("fft", run_fft_benchmarks),
        ("linalg", run_linalg_benchmarks),
        ("optimize", run_optimize_benchmarks),
        ("special", run_special_benchmarks),
        ("signal", run_signal_benchmarks),
        ("integrate", run_integrate_benchmarks),
        ("stats", run_stats_benchmarks),
        ("sparse", run_sparse_benchmarks),
    ]

    for group_name, group_func in benchmark_groups:
        if args.filter and args.filter not in group_name:
            continue
        try:
            results = group_func()
            for r in results:
                r.name = f"{group_name}/{r.name}"
            all_results.extend(results)
        except Exception as e:
            print(f"Error in {group_name}: {e}", file=__import__('sys').stderr)

    if args.json:
        output = {
            "benchmarks": [
                {
                    "name": r.name,
                    "mean_ns": r.mean_ns,
                    "median_ns": r.median_ns,
                    "p95_ns": r.p95_ns,
                    "p99_ns": r.p99_ns,
                    "min_ns": r.min_ns,
                    "max_ns": r.max_ns,
                    "runs": r.runs,
                }
                for r in all_results
            ]
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"{'Benchmark':<50} {'Mean':>12} {'Median':>12} {'P95':>12} {'P99':>12}")
        print("-" * 100)
        for r in all_results:
            def fmt_time(ns):
                if ns >= 1e9:
                    return f"{ns/1e9:.3f}s"
                elif ns >= 1e6:
                    return f"{ns/1e6:.3f}ms"
                elif ns >= 1e3:
                    return f"{ns/1e3:.3f}µs"
                return f"{ns:.0f}ns"
            print(f"{r.name:<50} {fmt_time(r.mean_ns):>12} {fmt_time(r.median_ns):>12} {fmt_time(r.p95_ns):>12} {fmt_time(r.p99_ns):>12}")


if __name__ == "__main__":
    main()
