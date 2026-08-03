# TDSE-Z Documentation — Sphinx configuration
import os

project = "TDSE-Z"
copyright = "2026, Dr. Zakaria Dahbi"
author = "Dr. Zakaria Dahbi"
version = "0.1.0"
release = "0.1.0"

extensions = [
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
]

import os
import re
import glob

def after_build(app, error):
    """Strip inline style attributes from HTML elements (RTD bakes dark backgrounds
    into inline styles which beat CSS !important), but keep theme.css for layout."""
    build_dir = app.outdir
    for html_file in glob.glob(os.path.join(build_dir, "**", "*.html"), recursive=True):
        with open(html_file, "r") as f:
            content = f.read()
        # Remove inline style attributes — they beat all CSS including !important
        content = re.sub(r'\s+style="[^"]*"', '', content)
        # Remove the JS that applies inline dark overrides
        content = content.replace('<script src="_static/js/theme.js"></script>', '')
        # Remove pygments dark theme CSS
        content = content.replace('<link rel="stylesheet" type="text/css" href="_static/pygments.css?v=b86133f3" />', '')
        with open(html_file, "w") as f:
            f.write(content)


def setup(app):
    app.connect('build-finished', after_build)

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "style": "neutral",
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
    "style_nav_header_background": "#2563eb",
}

# Custom CSS — dark glass theme overrides
html_css_files = [
    "custom.css",
]

# Custom JS for theme features
html_js_files = [
    "custom.js",
]

templates_path = ["_templates"]
html_static_path = ["_static"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_show_sourcemap = False
html_show_sphinx = False
html_show_copyright = True

# Favicon (remove if favicon doesn't exist)
# html_favicon = "_static/favicon.ico"

# MathJax configuration
mathjax_path = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'
mathjax3_config = {
    'tex': {
        'inlineMath': [['$', '$'], ['\\(', '\\)']],
        'displayMath': [['$$', '$$'], ['\\[', '\\]']],
        'processEscapes': True,
    },
    'options': {
        'skipHtmlTags': ['script', 'noscript', 'style', 'textarea', 'pre'],
    },
}
mathjax_config = mathjax3_config

# Toctree defaults
toc_object_entries = False
toc_backlink = "root"
suppress_warnings = ["toc.not_readable"]
