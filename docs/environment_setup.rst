Environment Setup
=================

Start with :doc:`quickstart` for a working API that uses a fake model and
SQLite. That example needs no provider credentials. This guide explains the
configuration for an application deployment.

Configuration sources
---------------------

For a source checkout, copy the template only if ``.env`` does not already
exist:

.. code-block:: bash

   cp .env.example .env

The template selects PostgreSQL and contains example model configurations.
Replace those values before starting the service. Configure only the features
that you use. A local fake-model service does not need PostgreSQL, LiteLLM,
Redis, or Langfuse.

The application reads environment variables and the nearest ``.env`` file.
Regular environment variables take precedence over values in that file.
After loading these settings, ``settings.setup()`` validates and applies
``LANGGRAPH_<NAME>`` environment overrides. These overrides take precedence
over the regular variables and the file. For example,
``LANGGRAPH_REQUEST_TIMEOUT`` overrides ``REQUEST_TIMEOUT``.

``PYTHON_DOTENV_DISABLED=1`` disables explicit ``load_dotenv()`` calls. It does
not stop Pydantic from reading the file selected by ``Settings``. In embedded
code, ``Settings(_env_file=None)`` disables file loading for that instance.
Repository tests and documentation builds also replace ``find_dotenv()`` to
prevent file discovery. The flag alone does not isolate a service from a local
``.env`` file.

Keep credentials in a secret manager or an untracked environment file. Do not
put real credentials in examples, source files, or test results. ``.gitignore``
excludes the root ``.env`` and the service environment files listed below.

Authentication and identity
---------------------------

For one trusted client application, configure:

.. code-block:: bash

   AUTH_MODE=trusted
   AUTH_SECRET=replace-with-a-private-shared-secret
   AUTH_SERVICE_USER_ID=service

The client sends ``Authorization: Bearer <AUTH_SECRET>``. In this mode, the
client application supplies ``user_id`` for its users. A missing ``user_id``
uses ``AUTH_SERVICE_USER_ID``. Keep the shared secret in that trusted
application. Use ``AUTH_MODE=token`` and ``AUTH_USERS`` when users connect to
the API directly. See :doc:`migration` for both request formats.

To accept feedback from token users, set ``FEEDBACK_SIGNING_SECRET`` to a
separate server-only random value with at least 32 characters. Use the same
value across workers and replicas. It must differ from every bearer token.
Trusted backend feedback does not require this setting. See :doc:`migration`
for response proof and client changes.

``user_id`` identifies a user for long-term memory. ``thread_id`` identifies
one conversation for short-term history. Setting ``user_id`` does not create a
long-term store. The application must configure and use that store.

Model providers
---------------

Install the provider extra before selecting a real model. See
:doc:`installation` for the available extras. The default OpenAI-backed
blueprints use these settings:

.. code-block:: bash

   USE_FAKE_MODEL=false
   OPENAI_MODEL_NAME=your-model-name
   OPENAI_API_BASE_URL=https://api.openai.com/v1

Supply ``OPENAI_API_KEY`` through the deployment environment. Choose a model
available to that account. A model request can incur provider charges.
``OPENAI_API_VERSION`` is not required for the public OpenAI endpoint.

For LiteLLM, set the model name to a configured proxy alias. Set the base URL to
``http://litellm:4000/v1`` from another Compose service, or
``http://127.0.0.1:4000/v1`` from the host. Supply the proxy key as
``OPENAI_API_KEY``. Container service names do not resolve on the host.

The API Docker image includes ``openai-aiohttp`` and sets
``LLM_HTTP_ASYNC_TRANSPORT=aiohttp``. Python installations default to ``httpx``.
Leave this setting unset in ``.env`` to keep the image or Python default. Set it
to ``httpx`` to override the image default. A Python installation needs the
``openai-aiohttp`` extra before selecting ``aiohttp``. See :doc:`reliability`
for pool limits and timeouts.

Named model configurations
--------------------------

``MODEL_CONFIGS`` maps application names to flat provider configurations:

.. code-block:: bash

   MODEL_CONFIGS='{"assistant":{"provider":"openai","name":"your-model-name"}}'

For a larger configuration, store JSON in a private file and set
``MODEL_CONFIGS_PATH``. For example, ``data/models.json`` can contain:

.. code-block:: json

   {
     "assistant": {
       "provider": "openai",
       "name": "your-model-name"
     },
     "azure_assistant": {
       "provider": "azure_openai",
       "name": "your-model-name",
       "azure_endpoint": "https://your-resource.openai.azure.com/",
       "azure_deployment": "your-deployment-name",
       "api_version": "your-supported-api-version"
     }
   }

.. code-block:: bash

   MODEL_CONFIGS_PATH=data/models.json

The factory accepts ``name`` or ``model_name``. It passes other fields to the
provider's model constructor. Use ``azure_endpoint`` and ``azure_deployment``
for Azure. The factory does not expand a nested ``params`` object. Supply
``OPENAI_API_KEY`` or ``AZURE_OPENAI_API_KEY`` through the environment for these
examples.

Set one configuration source. An explicit ``MODEL_CONFIGS`` value, including
``{}``, takes precedence over ``MODEL_CONFIGS_BASE64`` and ``MODEL_CONFIGS_PATH``.
Base64 takes precedence over the file path. Remove the inline template entry
before using the file path. Base64 encodes content; it does not encrypt it.

The ``chatbot-agent`` reads the request's ``model_config_key`` and creates the
selected model from this mapping. Custom agents must read that setting or use
``CompletionModelFactory.get_model_from_config`` themselves. The Deep Agent
blueprint instead selects ``MODEL_CONFIGS.deep_agent`` at startup. See
:doc:`integrations` and :doc:`deepagents` before changing model selection.

Persistence
-----------

For local durable conversation history, use SQLite:

.. code-block:: bash

   MEMORY_BACKEND=sqlite
   SQLITE_DB_PATH=checkpoints.db

The service must have write access to the database directory. A container needs
a persistent volume if history must survive container replacement. SQLite
checkpointing does not provide a long-term store.

For PostgreSQL, supply the password separately and configure:

.. code-block:: bash

   MEMORY_BACKEND=postgres
   POSTGRES_HOST=127.0.0.1
   POSTGRES_PORT=5432
   POSTGRES_USER=your-database-user
   POSTGRES_DB=agents
   POSTGRES_SCHEMA=public

Use ``POSTGRES_HOST=postgres`` for the repository's Compose network. Set
``POSTGRES_PASSWORD`` to the password for the selected user. Size checkpoint
and conversation-lock pools for the total worker count. See :doc:`deployment`
for database setup, pool sizing, and worker recovery.

Observability
-------------

Observability is optional. For Langfuse, install one SDK selector and set
``OBSERVABILITY_BACKEND=langfuse``. Supply ``LANGFUSE_PUBLIC_KEY``,
``LANGFUSE_SECRET_KEY``, and the project's ``LANGFUSE_HOST``.
``LANGFUSE_BASE_URL`` is an accepted URL alias and takes precedence over
``LANGFUSE_HOST``. Use ``http://langfuse-web:3000`` inside Compose or
``http://127.0.0.1:3000`` from the host.

For LangSmith, install ``langsmith`` and set
``OBSERVABILITY_BACKEND=langsmith``. Supply ``LANGSMITH_API_KEY``,
``LANGSMITH_PROJECT``, ``LANGSMITH_ENDPOINT=https://api.smith.langchain.com``,
and ``LANGSMITH_TRACING=true``. Keep the unused tracing backend disabled.
See :doc:`langfuse_compatibility` for the difference between SDK and server
versions.

Optional Compose services
-------------------------

The full Compose stack includes external databases, LiteLLM, and Langfuse.
Prepare its configuration files before starting it:

.. code-block:: bash

   cp configs/litellm/config.example.yaml configs/litellm/config.yaml
   cp configs/litellm/.litellm.env.example configs/litellm/.litellm.env
   cp configs/redis/.redis.env.example configs/redis/.redis.env
   cp configs/postgres/.postgres.env.example configs/postgres/.postgres.env
   cp configs/minio/.minio.env.example configs/minio/.minio.env
   cp configs/clickhouse/.clickhouse.env.example configs/clickhouse/.clickhouse.env
   cp configs/langfuse/.langfuse.env.example configs/langfuse/.langfuse.env

Use these copy commands only for files that do not yet exist. Replace example
passwords, host names, model aliases, and API versions for the selected
services. Keep shared credentials consistent across their environment files.
The LiteLLM example reads provider credentials through ``os.environ/...``
references. Supply the referenced variables to the LiteLLM container.

The checked-in Compose file selects Langfuse server v3. A package extra does
not change that server image. See :doc:`deployment` for startup commands and
:doc:`langfuse_compatibility` before changing the server version.
