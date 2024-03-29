# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

from sphinx_gallery.scrapers import matplotlib_scraper
from gf3d.plot.util import reset_mpl
project = 'GF3D'
copyright = '2023, Lucas Sawade'
author = 'Lucas Sawade'

# The full version, including alpha/beta/rc tags
release = '0.0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    "sphinx.ext.autosummary",
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'numpydoc',
    'sphinx_design',
    "sphinx_togglebutton",
    'sphinx_gallery.gen_gallery'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db",
                    ".DS_Store", "**.ipynb_checkpoints", "build"]


add_module_names = False
autoclass_content = 'both'
autosummary_generate = True

numpydoc_class_members_toctree = False

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'furo'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_css_files = ['css/custom.css']

html_logo = '_static/logo.png'


html_theme_options = {
    # "navbar_end": ["navbar-icon-links"],
    # "icon_links": [
    #     {
    #         # Label for this link
    #         "name": "GitHub",
    #         # URL where the link will redirect
    #         "url": "https://github.com/lsawade/GF3D",  # required
    #         # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
    #         "icon": "fa-brands fa-square-github",
    #         # The type of image to be used (see below for details)
    #         "type": "fontawesome",
    #     },
    # ]

}

html_favicon = '_static/favicon.ico'

html_context = {

    "default_mode": "auto",
    # "github_url": "https://github.com", # or your GitHub Enterprise site
    "github_user": "lsawade",
    "github_repo": "GF3D",
    "github_version": "main",
    "doc_path": "docs/source",
}

# --------------------------------


class matplotlib_svg_scraper(object):

    def __repr__(self):
        return self.__class__.__name__

    def __call__(self, *args, **kwargs):
        return matplotlib_scraper(*args, format='svg', **kwargs)


# Sphinx Gallery config
sphinx_gallery_conf = {
    # path to your example scripts
    'examples_dirs': [
        '../../examples/extraction/subset',
        '../../examples/extraction/database',
        '../../examples/generation',
        '../../examples/client'],
    # path to where to save gallery generated output
    'gallery_dirs': [
        "examples/extraction/subset",
        "examples/extraction/database",
        "examples/generation",
        'examples/client'],
    # Checks matplotlib for figure creation
    'image_scrapers': (matplotlib_svg_scraper(),),
    # Which files to include
    'filename_pattern': "/run_",

    'reset_modules': (reset_mpl, ),

    'remove_config_comments': True,
}
