#!/bin/bash
# Development environment setup script for GlassAlpha contributors
# This script creates a clean development environment with exact dependency versions

set -e  # Exit on any error

echo "🔧 Setting up GlassAlpha development environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.11"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python $required_version or higher required. Found: $python_version"
    exit 1
fi

echo "✅ Python version check passed: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade core tools
echo "⬆️  Upgrading pip, wheel, and build tools..."
pip install --upgrade pip wheel build

# Install dependencies with constraints
echo "📥 Installing dependencies with version constraints..."
pip install -c constraints.txt -e . --no-build-isolation

# Verify installation
echo "🧪 Verifying installation..."
python -c "import glassalpha; print(f'✅ GlassAlpha {glassalpha.__version__} installed successfully')"

# Install development tools
echo "🛠️  Installing development tools..."
pip install -c constraints.txt pytest pytest-cov pytest-mock mypy ruff black pre-commit

# Set up pre-commit hooks (if .pre-commit-config.yaml exists)
if [ -f ".pre-commit-config.yaml" ]; then
    echo "🪝 Installing pre-commit hooks..."
    pre-commit install
    echo "✅ Pre-commit hooks installed"
fi

# Run basic tests to verify everything works
echo "🧪 Running basic tests..."
python -m pytest tests/test_core_foundation.py -v

echo ""
echo "🎉 Development environment setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run tests:"
echo "  pytest"
echo ""
echo "To run linting:"
echo "  ruff check src/"
echo "  mypy src/"
echo ""
echo "To format code:"
echo "  black src/ tests/"
echo "  ruff check --fix src/"
