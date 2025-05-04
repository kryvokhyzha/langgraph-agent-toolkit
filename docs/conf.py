# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
import warnings

import rootutils
from sphinx_pyproject import SphinxConfig


# Set environment variables for fake models and authentication bypass
os.environ["USE_FAKE_MODEL"] = "true"
os.environ["OPENAI_API_KEY"] = "sk-fake-key-for-docs-generation"
os.environ["OPENAI_MODEL_NAME"] = "gpt-4-fake-model"
os.environ["OPENAI_API_BASE_URL"] = "https://fake-api.openai.com/v1"
os.environ["OPENAI_API_VERSION"] = "2023-05-15"
os.environ["LANGFUSE_SECRET_KEY"] = "lf-sk-fake-for-docs"
os.environ["LANGFUSE_PUBLIC_KEY"] = "lf-pk-fake-for-docs"
os.environ["LANGFUSE_HOST"] = "http://localhost:3000"
os.environ["MEMORY_BACKEND"] = "sqlite"
os.environ["SQLITE_DB_PATH"] = ":memory:"
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-fake-key"
os.environ["ANTHROPIC_MODEL_NAME"] = "claude-3-fake"
os.environ["GOOGLE_VERTEXAI_API_KEY"] = "fake-vertexai-key"
os.environ["GOOGLE_VERTEXAI_MODEL_NAME"] = "gemini-fake"
os.environ["GOOGLE_GENAI_API_KEY"] = "fake-genai-key"
os.environ["GOOGLE_GENAI_MODEL_NAME"] = "gemini-pro-fake"
os.environ["OBSERVABILITY_BACKEND"] = "empty"

# Find project root path - using pyproject.toml as indicator since .project-root might not exist
root_path = rootutils.find_root(search_from=__file__, indicator=["pyproject.toml"])
# Add project root to path so packages can be imported
rootutils.setup_root(root_path, indicator=["pyproject.toml"], pythonpath=True)

# Add the package to the path for autodoc to find it
sys.path.insert(0, os.path.abspath(root_path))

# Create a warning filter to ignore specific warnings during documentation building
warnings.filterwarnings("ignore", message=".*Model name must be provided for non-fake models.*")
warnings.filterwarnings("ignore", message=".*Missing required environment variables.*")
warnings.filterwarnings("ignore", message=".*Agent .* not found.*")
warnings.filterwarnings("ignore", message=".*unsupported operand type.*")
warnings.filterwarnings("ignore", message=".*has no attribute.*")

# Load configuration from pyproject.toml
config = SphinxConfig(os.path.join(root_path, "pyproject.toml"), globalns=globals())

# Explicitly set project information from pyproject.toml via SphinxConfig
project = config.name
author = "Roman Kryvokhyzha"
copyright = f"2023-2025, {author}"

# Extract version from pyproject.toml
release = config.version
version = ".".join(release.split(".")[:2])

# Additional project information from pyproject.toml
description = config.description
html_title = project

# Repository URLs
repository_url = "https://github.com/kryvokhyzha/langgraph-agent-toolkit"
documentation_url = "https://kryvokhyzha.github.io/langgraph-agent-toolkit"

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
    "sphinx.ext.githubpages",  # Enable linking to GitHub repository
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

# Intersphinx mappings - update with corrected URLs
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "langgraph": ("https://python.langchain.com/docs/integrations", None),  # Fixed URL
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
    "fastapi": ("https://fastapi.tiangolo.com/", None),
}


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

# Enable linking to GitHub repository
html_context = {
    "display_github": True,
    "github_user": "kryvokhyzha",
    "github_repo": "langgraph-agent-toolkit",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# Build documentation URL
html_baseurl = "https://kryvokhyzha.github.io/langgraph-agent-toolkit/"


# Function to resolve links to GitHub source code
def linkcode_resolve(domain, info):
    """Determine the URL corresponding to a Python object.

    This function links documentation to the source code on GitHub.
    """
    if domain != "py":
        return None

    modname = info["module"]
    fullname = info["fullname"]

    # Handle special cases like imported modules or objects
    if not modname:
        return None

    try:
        obj = sys.modules[modname]
        for part in fullname.split("."):
            obj = getattr(obj, part)

        # Get the source file
        import inspect

        try:
            source_file = inspect.getsourcefile(obj)
        except (TypeError, AttributeError):
            return None

        if source_file is None:
            return None

        # Convert source file path to relative path in the repository
        source_file = os.path.relpath(source_file, start=root_path)

        # Convert to URL
        # Line number info (if available)
        try:
            source_lines, lineno = inspect.getsourcelines(obj)
        except (OSError, TypeError):
            lineno = None

        if lineno:
            linespec = f"#L{lineno}-L{lineno + len(source_lines) - 1}"
        else:
            linespec = ""

        # Create GitHub URL
        github_url = f"{repository_url}/blob/main/{source_file}{linespec}"
        return github_url
    except Exception:
        return None
