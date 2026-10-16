# TDSE-Z Documentation — Sphinx configuration

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

# Numbered figures, tables and equations (:numref:, :eq:). Figures are
# pre-rendered SVGs in figures/ (sources and scripts next to them).
numfig = True
numfig_format = {"figure": "Fig. %s", "table": "Table %s", "code-block": "Listing %s"}
math_numfig = True
math_eqref_format = "Eq. ({number})"

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
    "logo_only": True,
    "style_nav_header_background": "#ffffff",
}
html_logo = "_static/logo.png"
html_favicon = "_static/favicon.png"
html_title = "TDSE-Z documentation"
html_css_files = ["custom.css"]
html_static_path = ["_static"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_show_sourcemap = False
html_show_sphinx = False
html_show_copyright = True

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
