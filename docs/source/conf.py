# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'spoQC'
copyright = '2026, Florian Heyl, Ezgi Sen'
author = 'Florian Heyl, Ezgi Sen'
release = '0.0.1'

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "myst_nb",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "readme_doc_statement.md",
    "readme_header.md"
]

myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "amsmath",
    "deflist",
    "fieldlist",
    "html_admonition",
    "html_image",
]

nb_execution_mode = "force"  # Re-run notebooks on every docs build
nb_execution_timeout = 600  
nb_execution_raise_on_error = True  # Fail docs build if any notebook cell errors
nb_remove_code_source = False  # Keep code cells visible by default in the rendered output

# Specify the BibTeX file for citations
bibtex_bibfiles = ["references.bib"]

# Intersphinx configuration
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "biocypher": ("https://biocypher.org/", None),
}

# Autodoc configuration
autodoc_typehints = "signature"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": True,
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
# html_extra_path = ["extra_files"]
# html_favicon = "_static/favicon.ico"
html_title = project

html_theme_options = {
    "repository_url": "https://github.com/heylf/spoQC",
    "use_repository_button": True,
    "use_download_button": True,
    "use_fullscreen_button": True,
    "navigation_with_keys": False,
}

autosummary_generate = True
autodoc_member_order = "groupwise"
default_role = "literal"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_use_rtype = True
napoleon_use_param = True
napoleon_use_ivar = True
napoleon_custom_sections = [("Params", "Parameters")]