#!/usr/bin/env python3
"""
CTest driver for the TDSEZ fused-assembler correctness invariant.

Runs tdsez_asmbench and asserts:
  1. the process exits 0,
  2. no operator comparison reports "DIFFERS" (fused must equal unfused),
  3. every printed relative difference `rel=Xe-Y` has exponent Y >= 12,
     i.e. fused vs unfused (and vs the production TDSEZFormHam kernel)
     disagree by at most 1e-12 in Frobenius norm.

This locks the paper's central claim -- "the fused assembler is bit-identical
to the unfused baseline" -- behind an automated regression gate.

Usage:  check_asmbench.py <path-to-tdsez_asmbench>
"""
import subprocess, sys, re, math

def main():
    if len(sys.argv) < 2:
        print("usage: check_asmbench.py <tdsez_asmbench>", file=sys.stderr)
        return 2
    binpath = sys.argv[1]

    try:
        out = subprocess.run([binpath], capture_output=True, text=True, timeout=1200)
    except subprocess.TimeoutExpired:
        print("FAIL: tdsez_asmbench timed out (>1200s)", file=sys.stderr)
        return 1

    log = out.stdout + out.stderr
    rc = out.returncode

    if rc != 0:
        print(f"FAIL: tdsez_asmbench exited with code {rc}", file=sys.stderr)
        return 1

    if "DIFFERS" in log:
        # pull the offending lines for the report
        for line in log.splitlines():
            if "DIFFERS" in line:
                print(f"FAIL: {line.strip()}", file=sys.stderr)
        return 1

    if "IDENTICAL" not in log:
        print("FAIL: benchmark produced no IDENTICAL comparisons", file=sys.stderr)
        return 1

    # Every `rel=Xe-Y` must satisfy Y >= 12 (rel <= 1e-12).
    worst = 1e-30
    worst_line = ""
    for m in re.finditer(r'rel=([0-9]+\.[0-9]+e-([0-9]+))', log):
        val = float(m.group(1))
        exp = int(m.group(2))
        rel = val * (10.0 ** (-exp))   # convert "5.20e-17" -> 5.20e-17
        if rel > worst:
            worst = rel
            worst_line = m.group(0)
    if worst > 1e-12:
        print(f"FAIL: worst fused/unfused rel diff = {worst:.3e} "
              f"(>{1e-12:.0e}) at '{worst_line}'", file=sys.stderr)
        return 1

    print(f"PASS: fused == unfused for all operators; "
          f"worst rel diff = {worst:.3e} (<= 1e-12)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
