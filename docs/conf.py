# Configure the Sphinx documentation builder.
# See https://www.sphinx-doc.org/en/master/usage/configuration.html for built-in values.

# Project information
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import ast
import inspect
import os
import sys
from datetime import date
from pathlib import Path

import dotenv
import rootutils
from sphinx import addnodes
from sphinx_pyproject import SphinxConfig


root_path = Path(rootutils.find_root(search_from=__file__, indicator=["pyproject.toml"]))

# Autodoc must not read local credentials or use deployment settings.
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
dotenv.find_dotenv = lambda *args, **kwargs: ""
dotenv.load_dotenv = lambda *args, **kwargs: False
settings_source = root_path / "langgraph_agent_toolkit/core/_base_settings.py"
settings_fields = set()
for definition in ast.parse(settings_source.read_text()).body:
    if isinstance(definition, ast.ClassDef) and definition.name == "Settings":
        for field in definition.body:
            if isinstance(field, ast.AnnAssign) and isinstance(field.target, ast.Name):
                settings_fields.add(field.target.id)
for name in tuple(os.environ):
    if name.upper() in settings_fields or name.upper().startswith(
        (
            "LANGGRAPH_",
            "LANGFUSE_",
            "LANGSMITH_",
            "LANGCHAIN_",
            "OTEL_",
            "OPENAI_",
            "AZURE_",
            "ANTHROPIC_",
            "GOOGLE_",
            "AWS_",
        )
    ):
        os.environ.pop(name)
os.environ.update(
    ENV_MODE="development",
    USE_FAKE_MODEL="true",
    OPENAI_API_KEY="documentation-test-only",
    OPENAI_MODEL_NAME="documentation-model",
    MEMORY_BACKEND="sqlite",
    SQLITE_DB_PATH=":memory:",
    OBSERVABILITY_BACKEND="empty",
    LANGSMITH_TRACING="false",
    LANGCHAIN_TRACING_V2="false",
    LANGFUSE_TRACING_ENABLED="false",
)
rootutils.setup_root(root_path, indicator=["pyproject.toml"], pythonpath=True, dotenv=False)
sys.path.insert(0, str(root_path))

# Load configuration from `pyproject.toml`.
config = SphinxConfig(os.path.join(root_path, "pyproject.toml"), globalns=globals())

# Set project information from `pyproject.toml`.
project = config.name
author = "Roman Kryvokhyzha"
copyright = f"2023-{date.today().year}, {author}"

# Extract the version from `pyproject.toml`.
release = config.version
version = ".".join(release.split(".")[:2])

# Load additional project information from `pyproject.toml`.
description = config.description
html_title = project

# Repository URLs
repository_url = "https://github.com/kryvokhyzha/langgraph-agent-toolkit"
documentation_url = "https://kryvokhyzha.github.io/langgraph-agent-toolkit"

# General configuration
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.linkcode",  # Add links to source code.
    "sphinx.ext.githubpages",  # Enable GitHub Pages links.
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_param = True  # Show parameter types and descriptions.
napoleon_use_rtype = True  # Show return types.

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": True,
    "special-members": "__init__",
    "show-inheritance": True,
    "ignore-module-all": True,
}
# Third-party inherited docstrings can use a different markup format.
autodoc_inherit_docstrings = False
autodoc_typehints = "description"
autoclass_content = "both"
autodoc_preserve_defaults = True  # Preserve default values in signatures.

# Enable autosummary.
autosummary_generate = True

# HTML output options
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {
    "logo_only": False,
    "style_external_links": True,
}

# Set the master document.
master_doc = "index"

# Show source links for all entities.
html_show_sourcelink = True

# Enable links to the GitHub repository.
html_context = {
    "display_github": True,
    "github_user": "kryvokhyzha",
    "github_repo": "langgraph-agent-toolkit",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# Set the documentation URL.
html_baseurl = "https://kryvokhyzha.github.io/langgraph-agent-toolkit/"


def qualify_builtin_types(app, doctree):
    """Keep the built-in type separate from message fields named type."""
    for node in doctree.findall(addnodes.pending_xref):
        if node.get("refdomain") == "py" and node.get("reftype") == "class" and node.get("reftarget") == "type":
            node["reftarget"] = "builtins.type"


def setup(app):
    """Resolve type annotations before Sphinx builds cross-references."""
    app.connect("doctree-read", qualify_builtin_types)


# Resolve links to GitHub source code.
def linkcode_resolve(domain, info):
    """Return the GitHub source URL for a Python object."""
    if domain != "py":
        return None

    modname = info["module"]
    fullname = info["fullname"]

    # Skip imported modules and objects.
    if not modname:
        return None

    try:
        obj = sys.modules[modname]
        for part in fullname.split("."):
            obj = getattr(obj, part)

        # Get the source file.
        try:
            source_file = inspect.getsourcefile(obj)
        except (TypeError, AttributeError):
            return None

        if source_file is None:
            return None

        # Convert the source path to a repository-relative path.
        source_path = Path(source_file).resolve()
        if not source_path.is_relative_to(root_path / "langgraph_agent_toolkit"):
            return None
        source_file = source_path.relative_to(root_path).as_posix()

        # Get line information when available.
        try:
            source_lines, lineno = inspect.getsourcelines(obj)
        except (OSError, TypeError):
            lineno = None

        if lineno:
            linespec = f"#L{lineno}-L{lineno + len(source_lines) - 1}"
        else:
            linespec = ""

        # Create the GitHub URL.
        github_url = f"{repository_url}/blob/main/{source_file}{linespec}"
        return github_url
    except Exception:
        return None
