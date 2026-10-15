#!/usr/bin/env python3
"""Strong-scaling table of the preconditioner benchmark (reads */run.log).

    python3 analyze.py [--plot scaling.pdf]

Per run: ranks, average GMRES iterations per solve, preconditioner set-up time
(PCSetUp, max over ranks), solve time per step (KSPSolve in the propagation
stage / steps), wall time per step, and speed-up / parallel efficiency of the
solve against the smallest rank count of the same deck and preconditioner.
"""
import argparse, glob, os, re
from collections import defaultdict


def parse(log):
    txt = open(log, errors="replace").read()
    its = [int(m) for m in re.findall(r"Linear \S* ?solve converged due to \S+ iterations (\d+)", txt)]
    div = len(re.findall(r"DIVERGED", txt))
    step = re.search(r"Wall time per step\s*=\s*([\d.eE+-]+)", txt)
    nsteps = re.search(r"Time steps\s*=\s*(\d+)", txt)
    # log_view: the propagation stage holds the time-stepping solves
    stage = txt.split("--- Event Stage")
    def event(name, text):
        vals = [float(m) for m in re.findall(rf"^{name}\s+\d+\s+[\d.]+\s+([\d.eE+-]+)", text, re.M)]
        return sum(vals)
    prop = next((s for s in stage if s.lstrip().startswith("1: TDSEZ Propagation")), "")
    return dict(its=sum(its) / len(its) if its else float("nan"), diverged=div,
                step=float(step.group(1)) if step else float("nan"),
                setup=event("PCSetUp", prop), solve=event("KSPSolve", prop),
                nsteps=int(nsteps.group(1)) if nsteps else 0, failed="ERROR" in txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot")
    args = ap.parse_args()
    runs = defaultdict(dict)
    for log in glob.glob("*_np*/run.log"):
        m = re.match(r"(.+)_(asm|mg|gamg)_np(\d+)$", os.path.basename(os.path.dirname(log)))
        if m:
            runs[(m.group(1), m.group(2))][int(m.group(3))] = parse(log)
    print(f"{'deck':12s} {'pc':5s} {'np':>5s} {'its':>6s} {'setup s':>8s} {'solve s/step':>13s} "
          f"{'wall s/step':>12s} {'speed-up':>9s} {'eff.':>6s}")
    for (deck, pc), byn in sorted(runs.items()):
        n0 = min(byn)
        for n in sorted(byn):
            r = byn[n]
            sps = r["solve"] / max(r["nsteps"], 1)
            ref = byn[n0]["solve"] / max(byn[n0]["nsteps"], 1)
            su = ref / sps if sps > 0 else float("nan")
            eff = su * n0 / n
            flag = "  FAILED" if r["failed"] or r["diverged"] else ""
            print(f"{deck:12s} {pc:5s} {n:5d} {r['its']:6.1f} {r['setup']:8.2f} {sps:13.4f} "
                  f"{r['step']:12.4f} {su:9.2f} {eff:6.2f}{flag}")
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        decks = sorted({d for d, _ in runs})
        fig, axes = plt.subplots(1, len(decks), figsize=(5 * len(decks), 4), squeeze=False)
        for ax, deck in zip(axes[0], decks):
            for (d, pc), byn in sorted(runs.items()):
                if d != deck:
                    continue
                ns = sorted(byn)
                ax.loglog(ns, [byn[n]["solve"] / max(byn[n]["nsteps"], 1) for n in ns], "o-", label=pc)
            ax.set_title(deck); ax.set_xlabel("MPI ranks"); ax.set_ylabel("solve time per step (s)")
            ax.legend()
        fig.tight_layout(); fig.savefig(args.plot)
        print("wrote", args.plot)


if __name__ == "__main__":
    main()
