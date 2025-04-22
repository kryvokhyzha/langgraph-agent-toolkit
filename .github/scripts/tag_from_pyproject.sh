#!/bin/bash
set -e

VERSION=$(python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])")
TAG="v$VERSION"

echo "Tagging current commit with: $TAG"
git tag $TAG
git push origin $TAG
