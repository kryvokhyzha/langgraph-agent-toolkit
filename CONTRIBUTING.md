# Contributing to Langgraph Agent Toolkit

First off, thank you for considering contributing to `Langgraph Agent Toolkit`!

## Development Setup

1. Make sure you have Python 3.10+ installed
2. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) package
   manager
3. Fork the repository
4. Clone your fork

   ```bash
   git clone https://github.com/YOUR-USERNAME/langgraph-agent-toolkit.git
   cd langgraph-agent-toolkit

   # Add the upstream remote
   git remote add upstream https://github.com/kryvokhyzha/langgraph-agent-toolkit.git
   ```

5. Set up the development environment:

   ```bash
   uv sync
   ```

   That's it! The `uv sync` command will automatically create and use a virtual
   environment.

6. Install pre-commit hooks:

   ```bash
   uv run pre-commit install
   uv run pre-commit run
   ```

   Pre-commit hooks will automatically run checks (like ruff, formatting, etc.)
   when you make a commit, ensuring your code follows our style guidelines.

### Running Commands

You have two options for running commands:

1. **With the virtual environment activated**:

   ```bash
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

   # Then run commands directly
   pytest
   pre-commit run --all-files
   ```

2. **Without activating the virtual environment**:

   ```bash
   # Use uv run prefix for all commands
   uv run pytest
   uv run pre-commit run --all-files
   ```

Both approaches work - use whichever is more convenient for you.

> **Note:** For simplicity, commands in this guide are mostly written
> **without** the `uv run` prefix. If you haven't activated your virtual
> environment, remember to prepend `uv run` to all python-related commands and
> tools.

### Adding Dependencies

When adding new dependencies to the library:

1. **Runtime dependencies** - packages needed to run the application:

   ```bash
   uv add new-package
   ```

2. **Development dependencies** - packages needed for development, testing, or
   CI:

   ```bash
   uv add --group dev new-package
   ```

After adding dependencies, make sure to:

1. Test that everything works with the new package
2. Commit both `pyproject.toml` and `uv.lock` files:

   ```bash
   git add pyproject.toml uv.lock
   git commit -m "Add new-package dependency"
   ```

## Development Process

1. Fork the repository and set the upstream remote
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run the tests (`pytest`)
5. Run pre-commit checks across all files (`pre-commit run --all-files`)
6. Commit your changes (`git commit -m 'Add some amazing feature'`)
   - Note: pre-commit will automatically run during commit for changed files
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request. Make sure the Pull Request's base branch is
   [the original repository's](https://github.com/kryvokhyzha/langgraph-agent-toolkit/)
   `main` branch.

## Code Style

We use pre-commit hooks to ensure code quality. These hooks include:

- **ruff** for linting and formatting
- **codespell** for spelling checks
- **prettier** for formatting markdown and other non-Python files
- Other quality checks for YAML, JSON, etc.

Rather than running linting tools manually, we recommend using pre-commit:

```bash
# Run pre-commit checks on all files
pre-commit run --all-files

# Or let it run automatically during git commits
git commit -m "Your commit message"
```

## Testing

We use pytest for testing. Please write tests for any new features and ensure
all tests pass:

```bash
# Run all tests
pytest
```

## Pull Request Process

1. Ensure your code follows the style guidelines of the project
2. Update the README.md with details of changes if applicable
3. The versioning scheme we use is [SemVer](http://semver.org/)
4. Include a descriptive commit message
5. Your pull request will be merged once it's reviewed and approved

## Code of Conduct

Please note we have a code of conduct, please follow it in all your interactions
with the project.

- Be respectful and inclusive
- Be collaborative
- When disagreeing, try to understand why
- A diverse community is a strong community

## Questions?

Don't hesitate to open an issue if you have any questions about contributing to
Langgraph Agent Toolkit.
