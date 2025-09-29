# GlassAlpha - Wheel-first development workflow
.PHONY: smoke build install test lint clean hooks help

# Default target
help:
	@echo "GlassAlpha Development Commands:"
	@echo ""
	@echo "🔥 smoke      - Run wheel smoke test (validates 4 critical contracts)"
	@echo "📦 build      - Build wheel"
	@echo "📥 install    - Install wheel (for testing)"
	@echo "🧪 test       - Run full test suite"
	@echo "🔍 lint       - Run linting"
	@echo "🧹 clean      - Clean build artifacts"
	@echo "🪝 hooks      - Install git hooks (pre-commit + pre-push)"
	@echo ""
	@echo "Smart git hooks workflow:"
	@echo "1. make hooks           (one-time setup)"
	@echo "2. git commit          (pre-commit: smoke test if fragile areas changed)"
	@echo "3. git push            (pre-push: always runs smoke test)"

# Smoke test - the key guardrail against CI thrashing
smoke:
	@echo "🔥 Running wheel smoke test..."
	./scripts/wheel_smoke.sh

# Build wheel
build:
	python3 -m build

# Install built wheel
install: build
	python3 -m pip install --force-reinstall --no-deps dist/*.whl

# Test suite (requires dependencies)
test:
	pytest -q

# Linting
lint:
	ruff check .
	mypy src/

# Clean artifacts
clean:
	rm -rf dist/ build/ *.egg-info/
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

# Install git hooks (one-time setup)
hooks:
	@echo "🪝 Installing git hooks..."
	@mkdir -p .git/hooks
	@ln -sf ../../scripts/pre-commit .git/hooks/pre-commit
	@ln -sf ../../scripts/pre-push .git/hooks/pre-push
	@echo "   ✅ pre-commit hook: runs smoke test when fragile areas change"
	@echo "   ✅ pre-push hook: always runs smoke test before push"
	@echo ""
	@echo "Hooks installed! Your commits and pushes are now protected."

# Fast pre-push checklist (what the user suggested)
check: smoke
	@echo "✅ Smoke test passed"
	@echo ""
	@echo "Optional: Run full test suite with dependencies installed:"
	@echo "  pip install -e ".[dev]" && make test"
