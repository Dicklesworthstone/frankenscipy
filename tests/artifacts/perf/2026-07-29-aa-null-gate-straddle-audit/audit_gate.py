import re, sys, glob
files = sorted(set(glob.glob('tests/artifacts/perf/2026-07-2*/*.txt')+glob.glob('tests/artifacts/perf/2026-07-2*/raw/*.txt')))
rows=[]
for path in files:
    txt=open(path,errors='replace').read()
    for blk in txt.split('fixture=')[1:]:
        m_id=re.search(r'method=(\w+) side=(\d+) n=(\d+)',blk)
        no=re.search(r'NULL-ours A/A median=([\d.]+) ci95=\[([\d.]+),([\d.]+)\]',blk)
        ns=re.search(r'NULL-scipy A/A median=([\d.]+) ci95=\[([\d.]+),([\d.]+)\]',blk)
        mr=re.search(r'Incumbent ratio: SciPy / FrankenSciPy = ([\d.]+)x',blk)
        mg=re.search(r'worst_null_edge=([\d.]+) required=([\d.]+) ratio_ci=\[([\d.]+),([\d.]+)\].*?=> (.+)',blk)
        if not(m_id and no and ns and mr and mg): continue
        om,ol,oh=map(float,no.groups()); sm,sl,sh=map(float,ns.groups())
        edge,req,rlo,rhi=map(float,mg.groups()[:4]); verdict=mg.group(5).strip()
        rows.append(dict(f=path.split('/')[-2][:26],meth=m_id.group(1),side=int(m_id.group(2)),
            om=om,ol=ol,oh=oh,sm=sm,sl=sl,sh=sh,ratio=float(mr.group(1)),
            edge=edge,req=req,rlo=rlo,rhi=rhi,v=verdict))
print(f"{'artifact':<27}{'meth':<9}{'side':>4} {'ratio':>8} {'req':>7} {'o-strad':>8} {'s-strad':>8} {'|o_med-1|':>9} {'|s_med-1|':>9} {'margin':>8}")
vetoed=[];bias=[]
for r in rows:
    o_str = r['ol']<=1.0<=r['oh']; s_str = r['sl']<=1.0<=r['sh']
    ob=abs(r['om']-1.0)*100; sb=abs(r['sm']-1.0)*100
    dev = r['rlo']-1.0 if r['ratio']>1 else 1.0-r['rhi']
    marg = dev/(r['req']-1.0) if r['req']>1.0 else float('inf')
    print(f"{r['f']:<27}{r['meth']:<9}{r['side']:>4} {r['ratio']:>8.4f} {r['req']:>7.4f} "
          f"{('YES' if o_str else 'NO!'):>8} {('YES' if s_str else 'NO!'):>8} {ob:>8.3f}% {sb:>8.3f}% {marg:>7.1f}x")
    if not o_str or not s_str: vetoed.append(r)
    if ob>2.0 or sb>2.0: bias.append(r)
print(f"\ncells audited: {len(rows)}")
print(f"cells a CI-STRADDLE gate would have VETOED: {len(vetoed)}")
for r in vetoed: print(f"   {r['f']} {r['meth']} side={r['side']} ratio={r['ratio']:.4f} "
                       f"ours_ci=[{r['ol']:.6f},{r['oh']:.6f}] scipy_ci=[{r['sl']:.6f},{r['sh']:.6f}]")
print(f"cells FAILING the corrected rule's 3rd clause (null median within 2% of 1.0): {len(bias)}")
print(f"max |null median - 1| across all cells: "
      f"{max(max(abs(r['om']-1),abs(r['sm']-1)) for r in rows)*100:.3f}%  (clause allows 2.000%)")
