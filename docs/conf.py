# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sqlutilpy

# -- Project information -----------------------------------------------------

project = 'sqlutilpy'
copyright = '2009-2026, Sergey E. Koposov'
author = 'Sergey E. Koposov'
release = sqlutilpy.__version__
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'numpydoc',
]

# numpydoc already generates the parameter tables, so autodoc must not
# add its own summaries of class members
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False

autodoc_member_order = 'bysource'

templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
language = 'en'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'navigation_depth': 3,
}
html_static_path = []

# -- Options for other output formats ----------------------------------------

htmlhelp_basename = 'sqlutilpydoc'
latex_documents = [
    (master_doc, 'sqlutilpy.tex', 'sqlutilpy Documentation', author,
     'manual'),
]
man_pages = [(master_doc, 'sqlutilpy', 'sqlutilpy Documentation', [author], 1)]
texinfo_documents = [
    (master_doc, 'sqlutilpy', 'sqlutilpy Documentation', author, 'sqlutilpy',
     'Query SQL databases and get numpy arrays back.', 'Miscellaneous'),
]
