import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "N-body-dark-plasma"
author = "Ahmed Ali"
release = "1.0.0"
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.viewcode"]
html_theme = "sphinx_rtd_theme"
autodoc_mock_imports = ["dynesty", "corner"]
