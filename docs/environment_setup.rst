Environment Setup
===============

Creating Your ``.env`` File
--------------------------

1. Copy the example configuration:

   .. code-block:: bash

      cp .env.example .env

2. Configure these essential sections:

   **LLM API Configuration**

   .. code-block:: bash

      # OpenAI Settings
      OPENAI_API_KEY=sk-xxxxxxxxxxxxx
      OPENAI_API_BASE_URL=http://litellm:4000/v1      # Can be OpenAI API or LiteLLM proxy
      OPENAI_API_VERSION=2025-01-01
      OPENAI_MODEL_NAME=gpt-4o-mini

      # Azure OpenAI Settings (Optional)
      AZURE_OPENAI_API_KEY=
      AZURE_OPENAI_ENDPOINT=
      AZURE_OPENAI_API_VERSION=
      AZURE_OPENAI_MODEL_NAME=
      AZURE_OPENAI_DEPLOYMENT_NAME=

      # Anthropic Settings (Optional)
      ANTHROPIC_API_KEY=
      ANTHROPIC_MODEL_NAME=

   **Multi-Provider Model Configuration**

   .. code-block:: bash

      # Configure multiple models from different providers in a single environment variable
      MODEL_CONFIGS={\
        "gpt4o": {\
          "provider": "azure_openai",\
          "name": "gpt-4o",\
          "api_key": "azure-key-123",\
          "endpoint": "https://your-resource.openai.azure.com/",\
          "api_version": "2023-05-15",\
          "deployment": "gpt4o-deployment",\
          "temperature": 0.7\
        },\
        "gpt4o-mini": {\
          "provider": "azure_openai",\
          "name": "gpt-4o-mini",\
          "api_key": "azure-key-123",\
          "endpoint": "https://your-resource.openai.azure.com/",\
          "api_version": "2023-05-15",\
          "deployment": "gpt4o-mini-deployment"\
        },\
        "gemini": {\
          "provider": "google_genai",\
          "name": "gemini-pro",\
          "api_key": "google-key-123",\
          "temperature": 0.7\
        }\
      }

   This configuration allows you to:

   - Define multiple models with different providers in one place
   - Reference them by logical names in your application
   - Set provider-specific parameters for each model
   - Switch between models without changing code

   **Database Configuration**

   .. code-block:: bash

      POSTGRES_HOST=postgres
      POSTGRES_PORT=5432
      POSTGRES_USER=postgres
      POSTGRES_PASSWORD=postgres
      POSTGRES_DB=postgres
      POSTGRES_SCHEMA=public

   **Observability Configuration**

   .. code-block:: bash

      # Option 1: Langfuse
      LANGFUSE_HOST=http://langfuse-web:3000
      LANGFUSE_PUBLIC_KEY=lf-pk-1234567890
      LANGFUSE_SECRET_KEY=lf-sk-1234567890

      # Option 2: LangSmith
      LANGSMITH_TRACING=true
      LANGSMITH_API_KEY=api-key-xxxxxxxxcxxxxxx
      LANGSMITH_PROJECT=default
      LANGSMITH_ENDPOINT=https://api.smith.langchain.com

3. Customize other sections as needed (Redis, memory options, logging)

.. warning::
   The ``.env`` file contains sensitive credentials and should never be committed
   to version control. It's already included in ``.gitignore``.

LiteLLM Configuration
--------------------

1. Create your configuration:

   .. code-block:: bash

      cp configs/litellm/config.example.yaml configs/litellm/config.yaml

2. Edit ``configs/litellm/config.yaml`` to include your models and credentials:

   **model_list**: Define available models

   .. code-block:: yaml

      model_list:
        - model_name: gpt-4o-mini # This name is used when selecting the model in your app
          litellm_params:
            model: azure/gpt-4o-mini # Format for the underlying LiteLLM model
            litellm_credential_name: your_azure_credential # References credentials defined below
            rpm: 6 # Rate limit (requests per minute)

   **credential_list**: Store API keys and endpoints

   .. code-block:: yaml

      credential_list:
        - credential_name: your_azure_credential
          credential_values:
            api_key: "your-api-key-here" # Best practice: use os.environ/AZURE_API_KEY
            api_base: "https://your-azure-endpoint.openai.azure.com/"
            api_version: "2025-01-01-preview"

   **router_settings**: Configure routing

   .. code-block:: yaml

      router_settings:
        routing_strategy: simple-shuffle
        redis_host: redis
        redis_password: os.environ/REDIS_AUTH

3. Setup service environment files:

   .. code-block:: bash

      cp configs/litellm/.litellm.env.example configs/litellm/.litellm.env
      cp configs/redis/.redis.env.example configs/redis/.redis.env
      cp configs/postgres/.postgres.env.example configs/postgres/.postgres.env
      cp configs/minio/.minio.env.example configs/minio/.minio.env
      cp configs/clickhouse/.clickhouse.env.example configs/clickhouse/.clickhouse.env
      cp configs/langfuse/.langfuse.env.example configs/langfuse/.langfuse.env

   For example, edit LiteLLM environment:

   .. code-block:: bash

      LITELLM_MASTER_KEY=sk-your-master-key  # Create a strong key here
      LITELLM_ENVIRONMENT=development
      DATABASE_URL="postgresql://postgres:postgres@postgres:5432/litellm"
      STORE_MODEL_IN_DB=True

.. note::
   LiteLLM relies on Redis for request caching and rate limiting.
