"""P1 check: did the MINRES reductions stop being one serial scalar chain?

Disassembles the `fsci_sparse::linalg::minres` body and reports, for each
floating-point add, which accumulator register it targets. A serial chain shows
one destination register for nearly every add; independent lanes show several.
"""
import re
import subprocess
import sys
from collections import Counter

binary = sys.argv[1]
label = sys.argv[2] if len(sys.argv) > 2 else binary

dis = subprocess.run(
    ["objdump", "-d", "-C", "--no-show-raw-insn", binary],
    capture_output=True, text=True, check=True,
).stdout

body, inside = [], False
for line in dis.splitlines():
    if re.search(r"<fsci_sparse::linalg::minres>:", line):
        inside = True
        continue
    if inside:
        if re.match(r"^[0-9a-f]+ <", line):
            break
        body.append(line)

mnemonics = Counter()
add_dests = Counter()
for line in body:
    m = re.search(r"\t(v[a-z0-9]+)\s+(.*)$", line)
    if not m:
        continue
    mnem, operands = m.group(1), m.group(2)
    if mnem.startswith(("vadd", "vmul", "vfmadd", "vsub")):
        mnemonics[mnem] += 1
    if mnem in ("vaddsd", "vaddpd"):
        dest = operands.split(",")[-1].strip()
        add_dests[dest] += 1

print(f"=== {label} : fsci_sparse::linalg::minres ({len(body)} insns) ===")
print("float mnemonics:", dict(mnemonics.most_common()))
print("add destinations:", dict(add_dests.most_common(8)))
packed = sum(v for k, v in mnemonics.items() if k.endswith("pd"))
scalar = sum(v for k, v in mnemonics.items() if k.endswith("sd"))
print(f"packed={packed} scalar={scalar} distinct_add_dests={len(add_dests)}")
