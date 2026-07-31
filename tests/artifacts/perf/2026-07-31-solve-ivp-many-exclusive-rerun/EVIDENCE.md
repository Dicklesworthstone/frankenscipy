# Exclusive `solve_ivp_many` completion sweep

## Result

This artifact closes `frankenscipy-eyr23` with two deliberately separate
findings:

1. The 32-thread whole-job cell is a **DECIDED FrankenSciPy win** against live
   SciPy 1.17.1: paired median SciPy / FrankenSciPy `464.488412x`, bootstrap
   median 95% CI `[432.012878, 481.458752]`. Whole-batch p50s were
   `1.892562 ms` and `877.111685 ms`.
2. The pre-registered post-16 worker-coordination mechanism is
   **FALSIFIED under its gate**. Zero of the three required comparisons
   cleared every clause, so `frankenscipy-ldx0f` must not ship a thread cap.

The second finding is not rescued by the first.

## Pre-registration and job

The mechanism and falsifier were committed before building or timing in
`6b1bc44ea`. The job was unchanged across every cell:

- 128 deterministic Lotka-Volterra trajectories;
- public `solve_ivp_many`, RK45, `t_span=[0,10]`;
- `rtol=1e-8`, `atol=1e-10`, `t_eval=None`;
- live SciPy 1.17.1 looping over the same 128 initial states;
- 11 interleaved A/B rounds and independent A/A controls per invocation;
- balanced order
  `1,128,2,64,4,32,8,16,16,8,32,4,64,2,128,1`;
- two occurrences pooled only after both raw logs were retained.

The deterministic bootstrap used seed `0x6a09e667f3bcc909`, 10,000 resamples,
and sorted indices 250 and 9750 for the 95% interval. The incumbent effect is
the median of paired per-round `SciPy / FrankenSciPy` ratios. Cross-cell
effects use an independent-bootstrap ratio of medians,
`median(cell) / median(16)`.

## Pooled incumbent results

Each row contains 22 paired rounds. Requested FrankenSciPy threads equaled
actual observed workers in every invocation; actual observed SciPy workers
were always one.

| Franken threads | Franken p50 ms/batch | SciPy p50 ms/batch | paired SciPy / Franken | bootstrap 95% CI | Franken null median | SciPy null median | corrected-gate result |
|---:|---:|---:|---:|---|---:|---:|---|
| 1 | 7.361757 | 865.501212 | 116.891930x | [114.616339, 118.600003] | 1.002258 | 1.002348 | WIN |
| 2 | 4.340388 | 865.270810 | 203.896970x | [189.184627, 206.876005] | 1.003623 | 0.990417 | WIN |
| 4 | 2.847499 | 864.983968 | 307.502212x | [199.925321, 343.115266] | 1.029442 | 0.993652 | NOT DECIDED: Franken null median |
| 8 | 2.233069 | 877.361627 | 390.349660x | [339.186284, 471.808262] | 1.002406 | 0.995520 | WIN |
| 16 | 1.777332 | 860.788662 | 478.922053x | [450.094053, 518.253399] | 1.023360 | 0.991527 | NOT DECIDED: Franken null median |
| 32 | 1.892562 | 877.111685 | 464.488412x | [432.012878, 481.458752] | 0.986495 | 1.002244 | WIN |
| 64 | 3.415286 | 873.427164 | 256.023652x | [250.279362, 270.165085] | 1.022216 | 1.002061 | NOT DECIDED: Franken null median |
| 128 | 6.163570 | 869.228672 | 140.344171x | [136.122844, 141.800892] | 1.017709 | 1.021362 | NOT DECIDED: SciPy null median |

For the headline 32-thread row, the Franken A/A null was median `0.986495`,
CI `[0.974161, 1.231786]`; the SciPy null was median `1.002244`, CI
`[0.992427, 1.019093]`. The effect CI excluded one, its deviation exceeded
twice the larger null half-width (`0.257626`) and the stricter endpoint margin
(`0.463572`), and both null medians were within 2% of one. Null-CI straddling
was telemetry only. Ratio CV was `14.503%`, recorded as provenance only.

## Pre-registered mechanism score

The observed wall-clock curve rose beyond 16, but the gate—not visual shape—
decides:

| comparison | ratio of Franken medians | bootstrap 95% CI | gate result |
|---|---:|---|---|
| 32 / 16 | 1.064833x | [0.973198, 1.184568] | not slower: effect CI includes one; 16-thread null median also fails |
| 64 / 16 | 1.921581x | [1.743597, 2.073227] | not decided: 16-thread null median is 1.023360 |
| 128 / 16 | 3.467878x | [3.179610, 3.790845] | not decided: 16-thread null median is 1.023360 |

The pre-registration required at least two decided slowdowns. The score is
`0/3`; therefore the mechanism is falsified and no production thread cap is
authorized.

## Scientific and execution proof

Every invocation completed all 128 trajectories at `t=10`. Both accepted-step
histories stayed finite and positive. All 256 final components were compared:
maximum absolute difference `6.573e-14`, maximum scaled difference below
`0.001`, and maximum Lotka invariant drift `1.212e-7` in both arms. Counted
work was nearly equal: FrankenSciPy `159,998` RHS evaluations versus SciPy
`160,126`; both stored `25,748` accepted points.

**Counted mechanism:** the work counts rule out a cheaper mathematical solve.
The measured boundary is 32 actual parallel FrankenSciPy workers executing
compiled solver stages and inline RHS calls versus one actual SciPy worker
running 128 serial public solves with Python solver-loop and callback tax.

The exclusive booking was Agent Mail CLAIM `7217`, RELEASE `7230`. The host
was `threadripperje`, AMD Ryzen Threadripper PRO 5995WX, 64 physical cores,
128 logical threads, `ram_bytes=536069869568`, one NUMA node, performance
governor, runtime AVX2/FMA/BMI2/VAES present and AVX-512F absent. Admission
found no benchmark or compiler process, 98.69% average idle, and zero iowait;
post-run sampling was 99-100% idle with zero iowait and no remaining arm.

The executed release ELF was built by strict remote RCH on `vmi1156319` from
base `6b1bc44ea` with `--clean-overlay --no-overlay` and a task-unique target.
It self-reported SHA-256
`54664c26480945aa63f338a89279fa6816f5c69eea126d7fefbc64edcba86161`.
The SciPy engine self-reported SHA-256
`aa16f42cc85fa02769ff00bf93bcdb48b6bf568e2d9f8ce48f9f378e76cf8f09`.
The raw capture is `bench_stdout_stderr.txt`, SHA-256
`f9220a1d593d071d30c5f2bf02095d0ddf49b9d83cbc0f3412633b80ef83d28c`.

## Retry and chooser

Do not repeat this unchanged scaling sweep. Reopen thread-count work only
after a production worker-lifecycle/pool change, a changed solver engine, or a
different named batch-size/per-trajectory-work regime. Preserve the same
balanced design, observed-thread proof, dual nulls, and corrected gate.

**CHOOSER STATEMENT:** for this exact 128-trajectory completion ensemble on a
32-CPU affinity, pick FrankenSciPy `solve_ivp_many`: it completed the whole
batch in `1.892562 ms` p50 versus `877.111685 ms` for a live SciPy 1.17.1
loop, a decided `464.488412x` ratio. Do not infer a general thread cap from
this result; the pre-registered scaling mechanism was falsified.
