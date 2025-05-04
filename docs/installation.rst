Installation Options
===================

The toolkit supports multiple installation options using "extras" to include
just the dependencies you need.

Basic Installation
-----------------

.. code-block:: bash

   pip install langgraph-agent-toolkit

Available Extras
---------------

LLM Provider Extras
^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # OpenAI, Uvicorn backend, and Langfuse observability
   pip install "langgraph-agent-toolkit[openai,uvicorn-backend,langfuse]"              
   
   # Anthropic, AWS Lambda backend, and Langsmith observability
   pip install "langgraph-agent-toolkit[anthropic,aws-backend,langsmith]"              

Additional Provider Options
^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``google-vertexai``
- ``google-genai``
- ``aws``
- ``ollama``
- ``groq``
- ``deepseek``

Full Installation
^^^^^^^^^^^^^^^^

.. code-block:: bash

   # All LLM providers, all backends, and all observability platforms
   pip install "langgraph-agent-toolkit[all-llms,all-backends,all-observability]"      

Client-Only Installation
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Just the client and Streamlit app
   pip install "langgraph-agent-toolkit[client]"
