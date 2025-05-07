# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
