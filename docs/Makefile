# Minimal Makefile for Sphinx documentation.
#

# Set these variables on the command line.
# You can also set the first two variables in the environment.
SPHINXOPTS    ?= -W --keep-going
SPHINXBUILD   ?= uv run --no-sync sphinx-build
SPHINXAPIDOC  ?= uv run --no-sync sphinx-apidoc
SOURCEDIR     = .
BUILDDIR      = _build

# Place this target first so `make` runs `make help` by default.
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help api html Makefile

# Generate the API root and remove pages for deleted modules.
api:
	@$(SPHINXAPIDOC) -f -e -M --remove-old -o generated ../langgraph_agent_toolkit --doc-project="API Reference"

html: api
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Send unknown targets to Sphinx with the `make` mode.
# `$(O)` is a shortcut for `$(SPHINXOPTS)`.
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
