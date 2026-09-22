LangGraph Agent Toolkit
=======================

Serve LangGraph agents through an authenticated HTTP API. Add streaming,
persistent conversation history, managed model connections, and optional tools
and observability. Use your own client or the included Streamlit interface.

Start here
----------

* Follow :doc:`quickstart` to run a local API without a model key.
* Follow :doc:`onboarding` to integrate the toolkit into an application.
* Use :doc:`integrations` to choose an agent pattern and tool integration.
* Read :doc:`migration` before upgrading an existing deployment to |release|.
* Use :doc:`deployment` and :doc:`reliability` to configure a deployed service.

The toolkit supports Python 3.11–3.14. Optional extras select providers,
observability SDKs, MCP tools, Deep Agents, and the UI. See :doc:`installation`.

Memory has two scopes. ``thread_id`` identifies short-term conversation state.
``user_id`` identifies the user scope for long-term stores. Passing a user ID
does not create a store. See :doc:`usage` and :doc:`integrations` for the contracts.

The running service exposes its HTTP schema at ``/docs`` and ``/openapi.json``.
The generated reference below describes the Python package.

.. toctree::
   :maxdepth: 2
   :caption: Start and configure

   quickstart
   onboarding
   installation
   environment_setup
   usage

.. toctree::
   :maxdepth: 2
   :caption: Agents and integrations

   integrations
   mcp
   deepagents
   langfuse_compatibility

.. toctree::
   :maxdepth: 2
   :caption: Deploy and upgrade

   deployment
   reliability
   migration
   dependency_updates
   related_projects

.. toctree::
   :maxdepth: 2
   :caption: Python API reference

   generated/modules

.. toctree::
   :maxdepth: 1
   :caption: Develop and verify

   contributing
   testing
   load_testing
   load_test_results
   live_llm_testing

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
