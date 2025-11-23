.. LangGraph Agent Toolkit documentation master file, created by
   sphinx-quickstart on Sun May  4 22:02:43 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

LangGraph Agent Toolkit Documentation
====================================

A comprehensive toolkit for building, deploying, and managing AI agents using
LangGraph, FastAPI, and Streamlit. It provides a production-ready framework for
creating conversational AI agents with features like multi-provider LLM support,
streaming responses, observability, and memory management.

What is langgraph-agent-toolkit?
--------------------------------

The langgraph-agent-toolkit is a full-featured framework for developing and
deploying AI agent services. Built on the foundation of:

- **LangGraph** for agent creation with advanced flows and human-in-the-loop capabilities
- **FastAPI** for robust, high-performance API services with streaming support
- **Streamlit** for intuitive user interfaces

Key components include:

- Data structures and settings built with **Pydantic**
- Multi-provider LLM support 
- Comprehensive memory management and persistence using PostgreSQL/SQLite
- Advanced observability tooling via Langfuse and Langsmith
- Modular architecture allowing customization while maintaining a consistent
  application structure

Whether you're building a simple chatbot or complex multi-agent system, this
toolkit provides the infrastructure to develop, test, and deploy your
LangGraph-based agents with confidence.

Architecture
-----------

.. image:: https://github.com/kryvokhyzha/langgraph-agent-toolkit/blob/main/docs/media/agent_architecture.png?raw=true
   :width: 800
   :alt: Architecture Diagram

Quickstart
----------

1. Create a ``.env`` file based on ``.env.example``

2. **Option 1: Run with Python from source**

   .. code-block:: bash

      # Install dependencies
      pip install uv
      uv sync --frozen
      source .venv/bin/activate

      # Start the service
      python langgraph_agent_toolkit/run_service.py

      # In another terminal
      source .venv/bin/activate
      streamlit run langgraph_agent_toolkit/streamlit_app.py

3. **Option 2: Run with Python from PyPi repository**

   .. code-block:: bash

      pip install langgraph-agent-toolkit

4. **Option 3: Run with Docker**

   .. code-block:: bash

      docker compose watch

Content
-------

.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   installation
   environment_setup
   usage
   
.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   generated/modules
   
.. toctree::
   :maxdepth: 1
   :caption: Development:

   contributing

Indices and tables
=================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
