uv_install_deps:
	uv sync --all-extras --no-install-project
uv_install_deps_compile:
	uv sync --all-extras --no-install-project --compile --no-cache
uv_get_lock:
	uv lock
uv_update_deps:
	uv sync --no-install-project --frozen
uv_update_self:
	uv self update
uv_show_deps:
	uv pip list
uv_show_deps_tree:
	uv tree
uv_build_wheel:
	uv build --wheel

# Install deployment dependencies (includes core dependencies + specified extras)
uv_install_deploy_all:
	uv sync --extras "deploy-all" --no-install-project

pre_commit_install: .pre-commit-config.yaml
	pre-commit install
pre_commit_run: .pre-commit-config.yaml
	pre-commit run --all-files
pre_commit_rm_hooks:
	pre-commit --uninstall-hooks

push_new_tag:
	sh .github/scripts/tag_from_pyproject.sh

run_langgraph:
	uv run langgraph dev