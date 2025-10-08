#!/bin/bash
# Start MkDocs development server with live reload
set -e

cd "$(dirname "$0")"

# Kill any existing MkDocs on port 8000
lsof -ti :8000 | xargs kill -9 2>/dev/null || true

# Activate venv and serve
source ../.venv/bin/activate
exec mkdocs serve --watch docs
