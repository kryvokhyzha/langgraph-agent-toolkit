# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.5]

### Fixed

- Downgrade langfuse

## [0.7.4]

### Updated

- Langfuse Callback import fix

## [0.7.3]

### Added

- New env variable for model config (base64)

## [0.7.2]

### Added

- New env variable for model config (file)

## [0.7.1]

### Fixed

- Agent executor test mock assertion to include additional `environment` and
  `tags` parameters
- Async test methods missing `@pytest.mark.asyncio` decorator in prompts tests

### Updated

- Dependencies to latest versions
- Agent executor `get_callback_handler` method to pass additional parameters:
  - Added `environment` parameter from settings.ENV_MODE
  - Added `tags` parameter with agent name for better observability tracking

### Improved

- Langfuse observability prompt hash detection with fallback mechanism:
  - Enhanced `push_prompt` method to use tags as fallback when commit_message is
    empty
  - Added robust `hasattr` checks for both `commit_message` and `tags`
    attributes
  - Improved logging to show old vs new hash values for better debugging

## [0.7.0]

### Fixed

- React Agent with SO

### Updated

- Dependencies
- Name of default configurable parameters

## [0.6.0]

### Fixed

- React Agent with SO

### Added

- Complex input

### Updated

- Dependencies
- Error handling
- Tests

## [0.5.0]

### Fixed

- Streamlit UI bugs
- Windows compatibility issue
- Enhance message handling inside `pre_hook_model`
- Add prompt hash to Langfuse observability class
- Rename few parameters

## [0.4.5]

### Fixed

- Strucuted output and model factory

## [0.4.4]

### Updated

- Project dependencies

### Fixed

- Strucuted output and model factory

## [0.4.3]

### Updated

- Project dependencies

## [0.4.2]

### Fixed

- Streaming bug
- Steamlit welcom message display
- Client handling error
- Package dependencies

## [0.4.1]

### Updated

- Client API to fully align with server endpoints
- Extended invoke, stream methods with additional parameters

### Added

- Message management methods in the client (add_messages, aadd_messages)
- Chat history retrieval methods (get_history, aget_history)
- History clearing methods (clear_history, aclear_history)
- Synchronous feedback creation method (create_feedback)
- Support for model_config_key parameter
- Support for recursion_limit parameter

### Fixed

- Client tests to properly mock API endpoints
- Parameter handling in stream and invoke methods

## [0.4.0]

### Updated

- Endpoints

### Added

- Endpoint to clear history
- Add new message to the history

### Fixed

- Minor fixes and refactoring

## [0.3.1]

### Added

- Ability to pass parameters to service runner
- Argument to select service runner

### Updated

- Service Dockerfile

## [0.3.0]

### Added

- `MODEL_CONFIGS` to unify LLM env variables
- New blueprint with AWS KB

### Fixed

- Streaming messages handling
- Refactored code structure for better maintainability
- Optional dependencies
- API exception handling

## [0.2.0]

### Fixed

- Refactored code structure for better maintainability
- Refactored Model factory

### Removed

- Removed AllModels and added environment variables for different providers

## [0.1.2]

### Added

- `user_id` parameter
- `store` creator to memory classes

### Fixed

- enhance error handling and testing in Streamlit app
- add new chat button
- variable names
- type hints

### Removed

- print statements

## [0.1.1]

### Added

- `get_default_agent` and `set_default_agent` functions

### Fixed

- Minor fixes
- Refactoring
- Update dependencies
- Update README

## [0.1.0]

### Changed

- Project structure
- Code style
- Agent blueprints

### Added

- Support of `Langfuse` observability platform.
- Agent executor
- Prompt manager
- Custom implementation of React Agent
- Service runners: standard, aws lambda, azure functions

### Fixed

- Minor fixes

### Removed

- Support of dozen LLM providers. They were replaced by a single one -
  `openai-compatible`. We can use `LiteLLM` as proxy for any LLM provider.
