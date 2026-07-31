#!/usr/bin/env python3
"""Re-address the LEDGER_RESURRECTION Appendix-A verdict index by content.

WHY THIS EXISTS
---------------
`docs/LEDGER_RESURRECTION.md` Appendix A records each hand-adjudicated REJECT
row as an *ordinal* into a positional screen:

    rg -in '^## .*(REJECT|INVALID|NO.SHIP|BLOCKER|dead.end)' \
      docs/NEGATIVE_EVIDENCE.md docs/progress/perf-negative-results.md | nl -ba

That ordinal is only meaningful against the exact file bytes the audit read,
which Appendix A pins by SHA-256. `docs/NEGATIVE_EVIDENCE.md` has since gained
rows, so the screen now returns 203 hits instead of 199 and EVERY ordinal past
the first insertion points at a different row. The narrow ISA table's
`NEGATIVE_EVIDENCE.md:<line>` citations rotted the same way.

Consequence: "re-run every row whose retry predicate is satisfied" was not
executable as written — the rows could not be identified. This script fixes the
addressing without touching a single verdict: it resolves each ordinal against
the ORIGINAL bytes (recovered from git), extracts the row's `## ` heading, and
relocates that heading in the CURRENT file. Headings are content-stable in this
ledger; ordinals are not.

Read-only with respect to the ledgers. Emits a remap table.

Run: python3 scripts/ledger_retry_remap.py [--json] [--class VOID-ISA]
"""

import json
import re
import subprocess
import sys

# The commit whose docs/NEGATIVE_EVIDENCE.md hashes to the SHA-256 Appendix A
# pins. Verified by hashing every candidate blob, not assumed.
AUDIT_COMMIT = '344c5102025aa25fd14c51f704dabc844f29812b'
AUDITED_SHA = {
    'docs/NEGATIVE_EVIDENCE.md':
        '86bd32d362edec962afebdc19e9be717d052a908cf31c7b4c5b1704c2c6c17af',
    'docs/progress/perf-negative-results.md':
        '0ca0a77545fe3ec0f80bdc028aa9f5b9c14e9c7eaa4b0a7e0e726ab6a3408f27',
}
FILES = list(AUDITED_SHA)
SCREEN = re.compile(r'^## .*(REJECT|INVALID|NO.SHIP|BLOCKER|dead.end)', re.I)

# Appendix A, verbatim.
CLASSES = {
    'VALID-AB': [1, 4, 8, 10, 106, 108, 109, 111, 112, 114, 143, 147, 148, 149,
                 154, 157, 161, 163, 195, 197],
    'VALID-PROFILE': [36, 48, 97, 105, 176, 186, 194],
    'VALID-MECHANISM': [6, 12, 15, 16, 18, 22, 23, 29, 39, 41, 49, 50, 51, 52,
                        53, 57, 59, 61, 67, 73, 78, 79, 81, 82, 86, 89, 93, 110,
                        113, 122, 130, 141, 146, 150, 151, 153, 162, 165, 169,
                        170, 171, 174, 175, 177, 178, 196, 199],
    'VOID-NONULL': [14, 17, 19, 20, 21, 24, 25, 26, 27, 31, 32, 33, 34, 37, 40,
                    43, 44, 45, 46, 47, 55, 56, 58, 60, 62, 63, 64, 65, 66, 68,
                    69, 70, 71, 74, 75, 80, 84, 91, 94, 98, 104, 115, 116, 117,
                    118, 119, 120, 121, 123, 124, 125, 126, 127, 131, 132, 133,
                    134, 135, 136, 137, 138, 139, 140, 142, 144, 145, 167, 168,
                    172, 173, 179, 180, 181, 182, 183, 187, 193],
    'VOID-CV': [5, 100, 101, 103, 158, 159, 160, 188, 189, 190, 192],
    'VOID-ZEROSELF': [95, 96, 184, 185],
    'VOID-ISA': [35, 42, 92],
    'EXCLUDED': [2, 3, 7, 9, 11, 13, 28, 30, 38, 54, 72, 76, 77, 83, 85, 87, 88,
                 90, 99, 102, 107, 128, 129, 152, 155, 156, 164, 166, 191, 198],
}


def read_at(commit, path):
    return subprocess.run(['git', 'show', f'{commit}:{path}'],
                          capture_output=True, text=True, check=True).stdout


def screen(texts):
    """Reproduce the `rg … | nl -ba` ordinal screen. 1-indexed."""
    hits = []
    for path, text in texts:
        for n, line in enumerate(text.split('\n'), 1):
            if SCREEN.match(line):
                hits.append({'file': path, 'line': n, 'heading': line.rstrip()})
    return hits


def main():
    audited = [(p, read_at(AUDIT_COMMIT, p)) for p in FILES]
    current = [(p, open(p, encoding='utf-8', errors='replace').read())
               for p in FILES]
    old, new = screen(audited), screen(current)

    # Heading -> current location. A heading repeated verbatim is ambiguous and
    # is reported as such rather than silently resolved to its first hit.
    index = {}
    for hit in new:
        index.setdefault(hit['heading'], []).append(hit)

    want = sys.argv[sys.argv.index('--class') + 1] if '--class' in sys.argv else None
    rows = []
    for klass, ordinals in CLASSES.items():
        if want and klass != want:
            continue
        for i in ordinals:
            if i > len(old):
                rows.append({'class': klass, 'ordinal': i, 'status': 'OUT-OF-RANGE'})
                continue
            src = old[i - 1]
            found = index.get(src['heading'], [])
            status = ('OK' if len(found) == 1
                      else 'MISSING' if not found else 'AMBIGUOUS')
            rows.append({
                'class': klass, 'ordinal': i, 'status': status,
                'heading': src['heading'],
                'audited_at': f"{src['file']}:{src['line']}",
                'current_at': (f"{found[0]['file']}:{found[0]['line']}"
                               if len(found) == 1 else None),
                'drifted': (len(found) == 1
                            and (found[0]['file'] != src['file']
                                 or found[0]['line'] != src['line'])),
            })

    if '--json' in sys.argv:
        print(json.dumps({'audit_commit': AUDIT_COMMIT,
                          'audited_hits': len(old), 'current_hits': len(new),
                          'rows': rows}, indent=2))
        return 0

    print(f'audited screen hits: {len(old)}   current screen hits: {len(new)}')
    drift = sum(1 for r in rows if r.get('drifted'))
    print(f'rows re-addressed: {len(rows)}   of which moved: {drift}')
    bad = [r for r in rows if r['status'] != 'OK']
    if bad:
        print(f'unresolvable by heading: {len(bad)}')
        for r in bad:
            print(f"  {r['status']:12} {r['class']:16} #{r['ordinal']} "
                  f"{r.get('heading', '')[:90]}")
    for r in rows:
        if r['status'] == 'OK' and (not want or r['class'] == want):
            mark = '->' if r['drifted'] else '=='
            print(f"{r['class']:16} #{r['ordinal']:3} {r['audited_at']:52} "
                  f"{mark} {r['current_at']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
