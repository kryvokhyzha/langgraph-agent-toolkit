# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import importlib.util
import inspect
import os
import sys
import types
import warnings
from unittest.mock import MagicMock

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
# Disable all observability
os.environ["OBSERVABILITY_BACKEND"] = "none"

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


# Advanced mocking for Python types that cause issues
class UnionTypeMock(MagicMock):
    """Special mock for Union types from typing."""

    def __or__(self, other):
        return MagicMock()

    def __ror__(self, other):
        return MagicMock()


# Create mocks for modules and types that are causing issues
MOCK_MODULES = [
    "langchain_core",
    "langchain_core.language_models.chat_models",
    "langchain_core.language_models",
    "langchain_core.messages",
    "langchain_core.prompts",
    "langchain_core.runnables",
    "langchain_core.tools",
    "langchain_community",
    "langgraph",
    "langgraph.graph",
    "langgraph.checkpoint",
    "langgraph.prebuilt",
    "langgraph_checkpoint_sqlite",
    "langgraph_checkpoint_postgres",
    "langgraph_supervisor",
    "joblib",
    "fastapi",
    "fastapi.middleware",
    "fastapi.responses",
    "streamlit",
    "pydantic",
    "pydantic.v1",
    "pydantic_settings",
    "langchain_openai",
    "langchain_anthropic",
    "langchain_google_vertexai",
    "langchain_google_genai",
    "langchain_ollama",
    "langchain_aws",
    "langchain_groq",
    "langchain_deepseek",
    "langfuse",
    "langsmith",
    "duckduckgo_search",
    "types.UnionType",
    "httpx",
    "uvicorn",
    "loguru",
    "asyncio",
    "jiter",
]

# Apply mocks
for mod_name in MOCK_MODULES:
    parts = mod_name.split(".")
    parent_mod = None

    # Build parent modules if needed
    for i in range(len(parts) - 1):
        parent_name = ".".join(parts[: i + 1])
        if parent_name not in sys.modules:
            sys.modules[parent_name] = MagicMock()
        parent_mod = sys.modules[parent_name]

    # Create the actual module
    if mod_name not in sys.modules:
        if mod_name == "types.UnionType":
            sys.modules[mod_name] = UnionTypeMock()
        else:
            sys.modules[mod_name] = MagicMock()

# Set up specific mocks for BaseChatModel and other complex types
sys.modules["langchain_core.language_models.chat_models.BaseChatModel"] = MagicMock()


# Create custom module with BaseChatModel
class BaseChatModelMock(MagicMock):
    pass


# Add BaseChatModel to langchain_core
if "langchain_core" in sys.modules:
    if not hasattr(sys.modules["langchain_core"], "language_models"):
        sys.modules["langchain_core"].language_models = MagicMock()
    if not hasattr(sys.modules["langchain_core"].language_models, "chat_models"):
        sys.modules["langchain_core"].language_models.chat_models = MagicMock()
    sys.modules["langchain_core"].language_models.chat_models.BaseChatModel = BaseChatModelMock

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
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Configure autodoc to be more forgiving of missing imports
autodoc_mock_imports = MOCK_MODULES

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


# Setup linkcode for source code links to GitHub
def linkcode_resolve(domain, info):
    """Link source code to GitHub repository."""
    if domain != "py" or not info["module"]:
        return None

    modname = info["module"]
    fullname = info["fullname"]

    # Check for mocked modules
    if modname in sys.modules and isinstance(sys.modules[modname], MagicMock):
        return None

    try:
        obj = sys.modules.get(modname)
        if obj is None:
            return None

        for part in fullname.split("."):
            try:
                obj = getattr(obj, part)
            except (AttributeError, TypeError):
                return None

        try:
            source_file = inspect.getsourcefile(obj)
            if source_file is None:
                return None
        except (TypeError, ValueError):
            return None

        # Create a clean relative path from project root
        source_file = os.path.relpath(source_file, start=os.path.dirname(os.path.abspath(root_path)))

        # GitHub URL pattern - use specific branch (main) and organization/repo
        github_url = f"{repository_url}/blob/main/{source_file}"

        try:
            source_lines, lineno = inspect.getsourcelines(obj)
            github_url += f"#L{lineno}-L{lineno + len(source_lines) - 1}"
        except (OSError, TypeError):
            pass

        return github_url
    except Exception:
        return None


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
