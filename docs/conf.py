# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "scikit-autoeval"
copyright = "2025, scikit-autoeval developers (BSD License)."
author = "Lucas Santos Rodrigues and Ismael Santana Silva"
release = "1.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_baseurl = "https://scikit-autoeval.github.io/scikit-autoeval/"

html_theme_options = {
    "navigation_with_keys": True,
    "light_logo": "logo-light.png",
    "dark_logo": "logo-dark.png",
    "sidebar_hide_name": True,
    "source_repository": "https://github.com/scikit-autoeval/scikit-autoeval/",
    "source_branch": "main",
    "source_directory": "docs/",
}

html_context = {
    "display_github": True,
    "github_user": "scikit-autoeval",
    "github_repo": "scikit-autoeval",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# html_theme_options["relbar"] = False
