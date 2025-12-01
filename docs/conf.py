# Configuration file for the Sphinx documentation builder.

import os
import sys
from datetime import datetime
from importlib.metadata import version

# ---------------------------------------------------------------------
# PATH SETUP — ensure autodoc finds the package in /src
# ---------------------------------------------------------------------
sys.path.insert(0, os.path.abspath("../src"))

# ---------------------------------------------------------------------
# PROJECT INFO
# ---------------------------------------------------------------------
project = "randomvar"
author = "Trevor Zabilowicz"

# Try to read package version from pyproject.toml
try:
    release = version("randomvar")
except Exception:
    release = "0.0.0"

# ---------------------------------------------------------------------
# GENERAL CONFIGURATION
# ---------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc.typehints",
]

autosummary_generate = True          # Automatically build API pages
autodoc_member_order = "bysource"    # Preserve source order
autodoc_typehints = "description"    # Show types in the description
autodoc_inherit_docstrings = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# ---------------------------------------------------------------------
# INTERSPHINX — link to other docs (Python stdlib)
# ---------------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", {}),
}

# ---------------------------------------------------------------------
# HTML OUTPUT
# ---------------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# ---------------------------------------------------------------------
# MASTER DOC — Critical for RTD
# ---------------------------------------------------------------------
master_doc = "index"
