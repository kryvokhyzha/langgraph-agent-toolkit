# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import inspect
import os
import sys

import rootutils
from sphinx_pyproject import SphinxConfig


# Find project root path
root_path = rootutils.find_root(search_from=__file__, indicator=[".project-root"])

# Add the package to the path for autodoc to find it
sys.path.insert(0, os.path.abspath(root_path))

# Load configuration from pyproject.toml
config = SphinxConfig(os.path.join(root_path, "pyproject.toml"), globalns=globals())

# Explicitly set project information from pyproject.toml via SphinxConfig
project = config.name  # Use name from pyproject.toml
author = "Roman Kryvokhyzha"  # Hardcoded as requested
copyright = f"2023-2025, {author}"

# Extract version from pyproject.toml
release = config.version
version = ".".join(release.split(".")[:2])

# Additional project information from pyproject.toml
description = config.description  # Use description from pyproject.toml
html_title = project

# URLs from pyproject.toml using SphinxConfig
repository_url = config.repository  # SphinxConfig provides this directly
documentation_url = config.documentation  # SphinxConfig provides this directly

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.linkcode",  # Add linkcode extension for better source links
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_param = True  # Show parameter types and descriptions
napoleon_use_rtype = True  # Show return types

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": True,
    "special-members": "__init__",
    "show-inheritance": True,
    "inherited-members": True,
}
autodoc_typehints = "description"
autoclass_content = "both"
autodoc_preserve_defaults = True  # Preserve default values in signature

# Enable autosummary
autosummary_generate = True

# Intersphinx mappings
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "langgraph": ("https://python.langchain.com/docs/integrations/langgraph", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
    "fastapi": ("https://fastapi.tiangolo.com/", None),
}


# Setup linkcode for source code links to GitHub
def linkcode_resolve(domain, info):
    """Link source code to GitHub repository."""
    if domain != "py" or not info["module"]:
        return None

    modname = info["module"]
    fullname = info["fullname"]

    obj = sys.modules.get(modname)
    if obj is None:
        return None

    for part in fullname.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None

    try:
        source_file = inspect.getsourcefile(obj)
        if source_file is None:
            return None
    except TypeError:
        return None

    # Create a clean relative path from project root
    source_file = os.path.relpath(source_file, start=os.path.dirname(os.path.abspath("../")))

    # GitHub URL pattern - use specific branch (main) and organization/repo
    github_url = f"{repository_url}/blob/main/{source_file}"

    try:
        source_lines, lineno = inspect.getsourcelines(obj)
        github_url += f"#L{lineno}-L{lineno + len(source_lines) - 1}"
    except (OSError, TypeError):
        pass

    return github_url


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
# html_logo = "_static/logo.png"  # Comment out logo as it doesn't exist
html_theme_options = {
    "logo_only": False,
    "display_version": True,
    "style_external_links": True,
}

# Set the master document
master_doc = "index"

# Show source links for all entities
html_show_sourcelink = True
