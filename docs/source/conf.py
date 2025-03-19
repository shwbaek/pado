import os
import sys
from datetime import datetime

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Project information
project = 'PADO Documentation'
copyright = f'{datetime.now().year}, Seung-Hwan Baek, Dong-Ha Shin, and contributors'
author = 'Seung-Hwan Baek, Dong-Ha Shin, and contributors'
release = '1.0.0'

# General configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx_autodoc_typehints',
    'sphinx_copybutton',
    'sphinx_design',
    'myst_parser',
    'nbsphinx',
]

# Try to import optional extensions
try:
    import sphinx_notfound_page
    extensions.append('sphinx_notfound_page')
except ImportError:
    pass

try:
    import sphinxext.opengraph
    extensions.append('sphinxext.opengraph')
except ImportError:
    pass

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
add_module_names = False

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}

# HTML output options
html_theme = 'furo'
html_title = 'PADO Documentation'
html_static_path = ['_static']
html_css_files = [
    'custom.css',
]
templates_path = ['_templates']

# Furo theme options
html_theme_options = {
    "announcement": "Pytorch Automatic Differentiable Optics",
    "sidebar_hide_name": False,
}

# Notebook execution settings
nbsphinx_execute = 'never'
nbsphinx_allow_errors = True

# MyST-Parser settings
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

# Copy button settings
copybutton_prompt_text = r">>> |\.\.\. |\$\ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True 