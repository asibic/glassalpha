#!/bin/bash
# GlassAlpha - Pre-commit Linting and Fixing Script
# Run this before any git commit to prevent pre-commit hook failures

set -e  # Exit on any error

echo "🔍 GlassAlpha - Pre-commit Linting & Auto-fix"
echo "=============================================="

# Navigate to packages directory
cd "$(dirname "$0")"

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

echo ""
echo "🧹 Step 1: Auto-fixing with Ruff..."
ruff check src/ tests/ --fix || {
    echo "❌ Some Ruff errors could not be auto-fixed. Please review and fix manually."
    echo "   Run: ruff check src/ tests/ --show-source"
    exit 1
}

echo ""
echo "🎨 Step 2: Formatting with Ruff..."
ruff format src/ tests/

echo ""
echo "⚫ Step 3: Formatting with Black (backup)..."
black src/ tests/ --quiet

echo ""
echo "🧪 Step 4: Running full pre-commit hooks..."
pre-commit run --all-files || {
    echo ""
    echo "❌ Pre-commit hooks failed! Issues have been automatically fixed where possible."
    echo "   Please review the changes and run this script again."
    exit 1
}

echo ""
echo "✅ All linting checks passed! Safe to commit."
echo ""
echo "💡 Next steps:"
echo "   git add ."
echo "   git commit -m 'Your commit message'"
echo "   git push"
