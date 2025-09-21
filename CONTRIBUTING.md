# Contributing to Glass Alpha

Thank you for your interest in contributing to Glass Alpha! This guide will help you get started with development and ensure your contributions align with our project goals.

## Project Overview

Glass Alpha is an open-source AI interpretability toolkit for tabular ML models.

## Development Setup

### Prerequisites
- Python 3.11+
- Git

### Local Development
```bash
# Clone and setup
git clone https://github.com/GlassAlpha/glassalpha
cd glassalpha
python -m venv .venv && source .venv/bin/activate

# Install package in development mode
pip install -e packages/glassalpha[dev]

# Run tests
pytest

# Setup documentation
cd site
pip install -r requirements.txt
mkdocs serve
```

## Code Standards

### Python Code Quality
- **Type hints everywhere**: We enforce `mypy --strict`
- **Linting**: Code must pass `ruff` and `black` formatting
- **Pre-commit hooks**: Required for all contributors
- **Test coverage**: 90% coverage required on core math utilities

### Code Structure
- Use src layout: `src/glassalpha/`
- Prefer functional, testable units over notebooks
- Notebooks live in `examples/` only
- All randomness must come from `glassalpha.utils.seeds`

### Testing Requirements
- **Unit tests**: `pytest` + `pytest-cov` 
- **Property tests**: Use `hypothesis` for determinism and constraints
- **Integration tests**: End-to-end CLI workflows with fixed seeds
- **Regression tests**: Byte-identical outputs for reproducibility
- **Performance benchmarks**: Track SHAP computation time on standard datasets

## Security & Privacy

- **No PII logging**: Sanitize and hash all IDs
- **Offline-first**: No outbound network calls in core library
- **Telemetry**: Gated behind `GLASSALPHA_TELEMETRY=on` (default off)
- **Reproducibility**: All runs write immutable manifests (config hash, dataset hash, git SHA, seeds)

## Documentation Requirements

Every contribution must include:

### Code Documentation
- Docstrings for all public APIs
- Type hints for all functions and methods
- Example usage in docstrings

### User Documentation  
- **Example notebook** in `examples/` for new features
- **API documentation** updates
- **README updates** if adding major features

### Content Guidelines
- All example notebooks must be runnable and produce expected outputs
- Include proper spellcheck compliance (see `site/docs/known_words.txt`)

## Contribution Process

### Before You Start
1. Check existing issues and discussions
2. For major changes, open an issue first to discuss the approach
3. Fork the repository and create a feature branch

### Pull Request Process
1. **Branch naming**: `feature/your-feature-name` or `fix/issue-description`
2. **Commit messages**: Use conventional commits format
3. **Testing**: Ensure all tests pass and coverage requirements are met
4. **Documentation**: Update relevant docs and add examples
5. **Review**: PRs require review from maintainers

### PR Checklist
- [ ] All tests pass (`pytest`)
- [ ] Code passes linting (`ruff`, `black`, `mypy --strict`)  
- [ ] Documentation updated (API docs + examples)
- [ ] Example notebook added for new features
- [ ] Security review completed (no PII logging, offline-compatible)
- [ ] Reproducibility verified (deterministic outputs with seeds)

## Contribution Guidelines
Please check our roadmap and open issues before starting major contributions

## Development Tips

### Config-First Development
- All CLIs must accept YAML config files
- Use configs/ for configuration definitions
- Validate policies at runtime

### Reproducibility Best Practices
- All randomness through centralized seeding
- Record seeds, git commit, and dataset hashes in outputs
- Ensure byte-identical outputs for identical inputs

### Performance Considerations
- Profile computation performance
- Optimize for typical dataset sizes
- Memory-efficient batch processing for large datasets

## Getting Help

- **Issues**: For bug reports and feature requests
- **Discussions**: For questions and design discussions
- **Documentation**: Check `site/docs/` for detailed guides

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## License

By contributing to Glass Alpha, you agree that your contributions will be licensed under the Apache-2.0 License.
