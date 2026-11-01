Contributing
============

Use :doc:`quickstart` to run the first API example. For source changes, read the
`contribution guide <https://github.com/kryvokhyzha/langgraph-agent-toolkit/blob/main/CONTRIBUTING.md>`_
and the repository's
`project rules <https://github.com/kryvokhyzha/langgraph-agent-toolkit/blob/main/CLAUDE.md>`_.

The package supports Python 3.11, 3.12, 3.13, and 3.14. Use Python 3.13 for development
because the pre-commit hooks require it. Install the locked dependencies with
``uv`` and select ``--extra all`` for all features. Choose at most one Langfuse
SDK version extra. See :doc:`installation` for these selections.

Use :doc:`testing` to choose the correct test layer. Local tests need no model
credentials. Process, container, and live-service checks have separate opt-in
flags. Add regression tests for changed behavior and run pre-commit before
submitting a pull request.

Write comments, docstrings, and documentation in ASD-STE100 Simplified Technical
English. Keep sentences short and preserve technical identifiers. Update the
relevant guide when an interface or configuration changes.

Build the documentation
-----------------------

Run these commands from the repository root. The installation includes the
locked documentation tools and optional features needed by the API reference:

.. code-block:: bash

   uv sync --frozen --no-install-project --extra all --group docs
   make -C docs html

The build regenerates ``docs/generated`` and builds the full documentation.
Sphinx warnings cause failure. Open ``docs/_build/html/index.html`` to review
the result. Edit package docstrings or source guides instead of generated pages.

On Windows, run the same sync command. Then run ``docs\make.bat html`` from
the repository root in Windows Command Prompt.

License
-------

The MIT License applies to this project. See the
`LICENSE file <https://github.com/kryvokhyzha/langgraph-agent-toolkit/blob/main/LICENSE>`_.
