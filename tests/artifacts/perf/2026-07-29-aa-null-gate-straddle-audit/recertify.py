import re,glob,sys
def cells():
    srcs=[('cpu63','lsqr_waived/lsqr_side_*.txt'),('cpu31sib','lsqr_repl_cpu31/lsqr_side_*.txt'),
          ('cpu15ind','lsqr_repl_cpu15/lsqr_side_*.txt')]
    srcs+=[('trj-gmres20','/data/projects/frankenscipy/tests/artifacts/perf/2026-07-29-sparse-nonsymmetric-vs-scipy-live-arm/gmres_restart20_bench_stdout_stderr.txt'),
           ('trj-orig','/data/projects/frankenscipy/tests/artifacts/perf/2026-07-29-sparse-nonsymmetric-vs-scipy-live-arm/bench_stdout_stderr.txt')]
    for lbl,pat in srcs:
        for p in sorted(glob.glob(pat)):
            t=open(p,errors='replace').read()
            for blk in t.split('fixture=')[1:]:
                mi=re.search(r'method=(\w+) side=(\d+)',blk)
                no=re.search(r'NULL-ours A/A median=([\d.]+) ci95=\[([\d.]+),([\d.]+)\]',blk)
                ns=re.search(r'NULL-scipy A/A median=([\d.]+) ci95=\[([\d.]+),([\d.]+)\]',blk)
                mg=re.search(r'worst_null_edge=([\d.]+) required=([\d.]+) ratio_ci=\[([\d.]+),([\d.]+)\].*?=> (.+)',blk)
                mr=re.search(r'FrankenSciPy = ([\d.]+)x',blk)
                if not(mi and no and ns and mg and mr): continue
                yield dict(run=lbl,meth=mi.group(1),side=int(mi.group(2)),
                    om=float(no.group(1)),ol=float(no.group(2)),oh=float(no.group(3)),
                    sm=float(ns.group(1)),sl=float(ns.group(2)),sh=float(ns.group(3)),
                    edge=float(mg.group(1)),req=float(mg.group(2)),
                    rlo=float(mg.group(3)),rhi=float(mg.group(4)),
                    ratio=float(mr.group(1)),legacy=mg.group(5).strip())
def cls(v): return 'WIN' if 'WIN' in v else ('LOSS' if 'LOSS' in v else 'INDET')
rows=list(cells())
print(f"{'run':<12}{'meth':<9}{'side':>4} {'ratio':>8} | {'LEGACY':>6} | {'c1':>3} {'c2':>3} {'c3':>3} {'CORRECTED':>9} | {'hw2x_req':>9} {'edge_req':>9} {'looser?':>7}")
tally={'became_decidable':[], 'became_vetoed':[], 'same':0}
for r in rows:
    # clause 1: effect CI excludes 1.0
    c1 = r['rlo']>1.0 or r['rhi']<1.0
    # clause 2: effect deviation (near edge of effect CI) > 2x LARGER null half-width
    hw = max((r['oh']-r['ol'])/2.0, (r['sh']-r['sl'])/2.0)
    dev = (r['rlo']-1.0) if r['ratio']>1 else (1.0-r['rhi'])
    c2 = dev > 2.0*hw
    # clause 3: each null MEDIAN within 2% of 1.0
    c3 = abs(r['om']-1.0)<=0.02 and abs(r['sm']-1.0)<=0.02
    corr = ('WIN' if r['ratio']>1 else 'LOSS') if (c1 and c2 and c3) else 'INDET'
    leg = cls(r['legacy'])
    hw2 = 1.0+2.0*hw
    looser = 'YES' if hw2 < r['req'] else 'no'
    print(f"{r['run']:<12}{r['meth']:<9}{r['side']:>4} {r['ratio']:>8.4f} | {leg:>6} | "
          f"{('Y' if c1 else 'N'):>3} {('Y' if c2 else 'N'):>3} {('Y' if c3 else 'N'):>3} {corr:>9} | "
          f"{hw2:>9.4f} {r['req']:>9.4f} {looser:>7}")
    if leg=='INDET' and corr!='INDET': tally['became_decidable'].append((r,corr))
    elif leg!='INDET' and corr=='INDET': tally['became_vetoed'].append((r,leg))
    else: tally['same']+=1
print(f"\ncells: {len(rows)}   unchanged: {tally['same']}")
bd=tally['became_decidable']
print(f"previously-vetoed rows that BECAME DECIDABLE: {len(bd)}"
      + (f"  -> WIN {sum(1 for _,c in bd if c=='WIN')}, LOSE {sum(1 for _,c in bd if c=='LOSS')}" if bd else ""))
for r,c in bd: print(f"   {r['run']} {r['meth']} side={r['side']} ratio={r['ratio']:.4f} -> {c}")
bv=tally['became_vetoed']
print(f"previously-decidable rows now VETOED by clause 3: {len(bv)}")
for r,l in bv: print(f"   {r['run']} {r['meth']} side={r['side']} ratio={r['ratio']:.4f} was {l}"
                     f"  |o_med-1|={abs(r['om']-1)*100:.2f}% |s_med-1|={abs(r['sm']-1)*100:.2f}%")
n_looser=sum(1 for r in rows if 1.0+2.0*max((r['oh']-r['ol'])/2,(r['sh']-r['sl'])/2) < r['req'])
print(f"\nclause-2 (2x half-width) is LOOSER than my existing 2x-endpoint margin in {n_looser}/{len(rows)} cells")
