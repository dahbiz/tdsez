# TDSE-Z Documentation — Sphinx configuration
import os, sys

project = "TDSE-Z"
copyright = "2025, Dr. Zakaria Dahbi"
author = "Dr. Zakaria Dahbi"
version = "0.1.0"
release = "0.1.0"

extensions = [
    "sphinx.ext.intersphinx",
]
intersphinx_mapping = {
    "petsc": ("https://petsc.org/release/", None),
}

# Custom HTML theme settings
html_theme = "basic"
html_sidebars = {}
html_copy_source = False
html_show_sourcemap = False
html_show_sphinx = False
html_show_copyright = True

html_css_files = [
    "custom.css",
]
