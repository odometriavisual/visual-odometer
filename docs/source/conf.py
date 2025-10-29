# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

# conf.py
import os
import sys

# Ensure current directory (where conf.py and doi_role.py live) is on sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'visual-odometer'
copyright = '2025, LASSIP'
author = 'LASSIP'
release = '0.4.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'doi_role',
    'sphinx_toolbox.more_autodoc',
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
    'sphinx.ext.imgmath',
    'sphinx.ext.viewcode',
    'sphinxcontrib.bibtex',
]

bibtex_bibfiles = ['references.bib']
bibtex_default_style = 'plain'

templates_path = ['_templates']
exclude_patterns = []

source_suffix = '.rst'
autoclass_content = 'both'

# The master toctree document.
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
