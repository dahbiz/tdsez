# TDSE-Z Documentation Assets

This directory contains static assets for the TDSE-Z project documentation and demo webpage.

## Contents

- `assets/` - Logo, favicon, CSS, JavaScript, and image assets for the docs site
- `h2p/` - High-harmonic generation / photoelectron spectrum visualizations (SVG, data, video)
- `waveform_*/` - Waveform demonstration directories

## Usage

These files are served as static content for the project website. They are **not** part of the C++ build.

## Generating plots

The Python scripts in `assets/` generate some of the visualization images:

```bash
python assets/generate_plots.py
python assets/generate_waveform_plots.py
```

Requires: matplotlib, numpy (see requirements in `docs/requirements.txt` if present).
