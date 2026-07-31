#!/usr/bin/env python3
"""Audit every KEEP claim in the repo against the fleet policy that a perf KEEP
requires a vs-incumbent ratio measured with the incumbent live in the same
invocation.

Read-only. Inventory only — this script never edits or weakens a claim.

CLAIM UNIT (stated so the partition is reproducible, not a judgement call):
  * docs/NEGATIVE_EVIDENCE.md — each `## ` entry whose decision line says KEEP
    (including RESURRECTED KEEP). This is the canonical ledger.
  * docs/perf_ledger_cc.md    — each `### ` section marked KEEP, plus each
    scorecard table row marked `✅ KEEP`. Scorecard rows are counted separately
    because many are per-workload cells of one underlying lever.

CLASSIFICATION, most-supported first. A claim lands in the first bucket it
matches:
  GATED_SAME_INVOCATION  an incumbent ratio AND explicit same-invocation live
                         incumbent evidence (same-invocation / side-by-side /
                         live SciPy arm / "Incumbent ratio:")
  RATIO_NOT_SAME_INVOC   an incumbent ratio, but measured against a separately
                         invoked or historical incumbent
  NO_INCUMBENT_RATIO     self-speedup / vs-serial / byte-identical Nx only
  NO_INCUMBENT_EXISTS    the surface has no incumbent to compare against at all
                         (a genuine gap-fill), which is a DIFFERENT problem from
                         nobody having measured it

Run: python3 scripts/keep_claim_gate_audit.py [--json]
"""

import json
import re
import sys

NEG = 'docs/NEGATIVE_EVIDENCE.md'
CC = 'docs/perf_ledger_cc.md'

# --- evidence markers -------------------------------------------------------
# NOTE: `[x×]\b` is WRONG here. `×` is a non-word character, so a trailing `\b`
# never matches `4.05× faster` (non-word followed by space is not a boundary)
# while it does match `4.05x faster`. That silently misclassified every claim
# written with the `×` glyph — most of the older ledger. Use a negative
# lookahead for a word/decimal continuation instead.
RE_RATIO = re.compile(r'\d[\d.,]*\s*[x×](?![\w.])')
RE_VS_INCUMBENT = re.compile(
    r'(faster|slower)\s+than\s+(live\s+)?(scipy|sklearn|numpy)'
    r'|scipy\s*/\s*frankenscipy'
    r'|incumbent ratio'
    r'|vs\.?\s+(live\s+)?(scipy|sklearn|numpy)'
    r'|(scipy|sklearn|numpy)\s+\S{0,24}(is|was)\s+[\d.]+\s*[x×]'
    r'|\|\s*[\d.]+\s*(ms|µs|us|s)\s*\|',   # scorecard: an incumbent time column
    re.I)
RE_SAME_INVOCATION = re.compile(
    r'same[- ]invocation|side[- ]by[- ]side same|same invocation'
    r'|legacy incumbent arm|live (scipy|sklearn) (1\.\d+\.\d+ )?arm'
    r'|Incumbent ratio:|live[- ]arm',
    re.I)
RE_SELF_ONLY = re.compile(
    r'self[- ]speedup|vs serial|forced[- ]serial|parallel vs serial', re.I)
# "No incumbent exists" means there is genuinely nothing to compare against.
# It is NOT the same as "scipy lacks this type but we still benchmarked against
# a scipy loop" — several gap-fill rows say "feature gap" in the title and then
# quote a scipy ratio in the body. Those have an incumbent. So this bucket
# requires an explicit no-peer statement AND the absence of any vs-incumbent
# ratio anywhere in the row.
RE_NO_INCUMBENT = re.compile(
    r'no (scipy|sklearn|numpy) (peer|equivalent|counterpart|analogue)'
    r'|(scipy|sklearn|numpy) has no\b'
    r'|no incumbent|not (present|available) in (scipy|sklearn)'
    r'|missing (from|in) (scipy|sklearn)',
    re.I)

# A row the ledger itself labels a self-speedup can never count as a gated
# incumbent win, however much live-arm context its prose mentions. Policy row
# docs/perf_ledger_cc.md:3560 already reclassified three BDF rows this way.
RE_DECLARED_SELF = re.compile(
    r'\*\*SELF-SPEEDUP\*\*|SELF-SPEEDUP|self[- ]speedup, not a campaign', re.I)


def classify(body: str) -> str:
    has_ratio = bool(RE_RATIO.search(body))
    vs_inc = bool(RE_VS_INCUMBENT.search(body))
    if RE_NO_INCUMBENT.search(body) and not vs_inc:
        return 'NO_INCUMBENT_EXISTS'
    # Only the row's own verdict counts. A row that merely *discusses* a
    # sibling self-speedup elsewhere in its prose is still a gated win, so look
    # at the heading and the decision region rather than the whole body.
    verdict_region = body[:body.find('\n') + 1] + body[:600]
    if RE_DECLARED_SELF.search(verdict_region):
        return 'NO_INCUMBENT_RATIO'
    if has_ratio and vs_inc:
        return ('GATED_SAME_INVOCATION' if RE_SAME_INVOCATION.search(body)
                else 'RATIO_NOT_SAME_INVOC')
    return 'NO_INCUMBENT_RATIO'


def sections(path, level):
    """Yield (line_no, heading, body) for each heading at `level`."""
    lines = open(path, encoding='utf-8', errors='replace').read().split('\n')
    marks = [i for i, l in enumerate(lines)
             if l.startswith(level + ' ') and not l.startswith(level + '#')]
    for idx, i in enumerate(marks):
        end = marks[idx + 1] if idx + 1 < len(marks) else len(lines)
        yield i + 1, lines[i], '\n'.join(lines[i:end])


def collect():
    claims = []
    for n, head, body in sections(NEG, '##'):
        decision = body[:2600]
        if re.search(r'\bKEEP\b', head) or re.search(
                r'Decision:\s*\**\s*(RESURRECTED\s+)?KEEP', decision):
            claims.append({'src': NEG, 'line': n, 'kind': 'ledger-entry',
                           'title': head.lstrip('# ').strip()[:140],
                           'bucket': classify(body)})
    for n, head, body in sections(CC, '###'):
        if 'KEEP' in head:
            claims.append({'src': CC, 'line': n, 'kind': 'ledger-entry',
                           'title': head.lstrip('# ').strip()[:140],
                           'bucket': classify(body)})
    lines = open(CC, encoding='utf-8', errors='replace').read().split('\n')
    for i, l in enumerate(lines):
        if l.startswith('|') and 'KEEP' in l:
            claims.append({'src': CC, 'line': i + 1, 'kind': 'scorecard-row',
                           'title': l.strip()[:140], 'bucket': classify(l)})
    return claims


def main():
    claims = collect()
    counts = {}
    for c in claims:
        counts[c['bucket']] = counts.get(c['bucket'], 0) + 1
    if '--json' in sys.argv:
        print(json.dumps({'total': len(claims), 'counts': counts,
                          'claims': claims}, indent=2))
        return 0

    total = len(claims)
    gated = counts.get('GATED_SAME_INVOCATION', 0)
    print(f'TOTAL KEEP CLAIMS: {total}')
    for b in ('GATED_SAME_INVOCATION', 'RATIO_NOT_SAME_INVOC',
              'NO_INCUMBENT_RATIO', 'NO_INCUMBENT_EXISTS'):
        n = counts.get(b, 0)
        print(f'  {b:24s} {n:>4}  ({100.0 * n / total:5.1f}%)')
    print(f'\nheadline: {gated}/{total} ({100.0 * gated / total:.1f}%) carry a '
          f'vs-incumbent ratio measured live in the same invocation; '
          f'{total - gated} do not.')

    for kind in ('ledger-entry', 'scorecard-row'):
        sub = [c for c in claims if c['kind'] == kind]
        sc = {}
        for c in sub:
            sc[c['bucket']] = sc.get(c['bucket'], 0) + 1
        print(f'\n  by unit — {kind}: {len(sub)} total, ' +
              ', '.join(f'{k}={v}' for k, v in sorted(sc.items())))
    return 0


if __name__ == '__main__':
    sys.exit(main())
