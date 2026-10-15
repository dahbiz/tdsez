# Documentation figures

The SVG files are committed so that the documentation builds with Sphinx alone
(the docs workflow installs only `sphinx` and `sphinx-rtd-theme`).

Regenerate them from the repository root:

```bash
cd docs/docs_sphinx/figures
for f in *.dot; do dot -Tsvg "$f" -o "${f%.dot}.svg"; done   # Graphviz diagrams
for f in make_*.py; do python3 "$f"; done                     # matplotlib plots (numpy, matplotlib)
```

The plots are computed from the formulas in the source code or from the
measured values quoted in the pages; none of them reads simulation output,
except `performance_propagators.svg`, which is produced by
`docs/perf/run.sh` (copy `docs/perf/propagator_dipoles.svg`).

If Graphviz reports `Format: "svg" not recognized`, its plugin configuration is
missing: run `dot -c` once (as the owner of the Graphviz installation), or point
`GVBINDIR` at a directory containing the plugins and a `config6` file.
