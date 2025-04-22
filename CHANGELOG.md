# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
