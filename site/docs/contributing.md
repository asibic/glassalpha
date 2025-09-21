# Contributing to Glass Alpha

Thank you for your interest in contributing to Glass Alpha! This guide focuses on **audit-first development**.

## Development Focus

!!! info "Contribution Guidelines"
    Contributions should focus on: **"Does this directly improve audit PDF quality, usability, or reproducibility?"**
    
    - Features should align with audit-first approach
    - Avoid complexity without clear user value

## Development Setup

### Prerequisites
- Python 3.11+
- Git
- Virtual environment tool (venv, conda, poetry)

### Local Development

1. **Fork and clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/glassalpha
cd glassalpha
```

2. **Create a virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install in development mode**
```bash
pip install -e packages/[dev]
pre-commit install
```

4. **Run tests to verify setup**
```bash
pytest
```

## Code Standards

### Type Hints
All code must have type hints and pass `mypy --strict`:
```python
def calculate_importance(
    values: np.ndarray,
    features: List[str],
    normalize: bool = True
) -> Dict[str, float]:
    """Calculate feature importance scores."""
    ...
```

### Code Quality
- **Formatting**: Code must pass `black` formatting
- **Linting**: Must pass `ruff` checks
- **Testing**: Maintain >90% coverage on core modules

### Testing Requirements
Write tests for all new features:
```python
def test_feature_importance_calculation():
    """Test that feature importance sums to 1 when normalized."""
    # Arrange
    values = np.array([0.1, 0.2, 0.3])
    features = ['f1', 'f2', 'f3']
    
    # Act
    result = calculate_importance(values, features, normalize=True)
    
    # Assert
    assert sum(result.values()) == pytest.approx(1.0)
```

## Development Priorities

### Audit PDF Quality
1. **Deterministic outputs**: Identical PDFs on same config/seed
2. **Professional formatting**: Publication-quality visualizations
3. **Complete lineage**: Git SHA, config hash, data hash tracking
4. **Error handling**: Graceful failures with clear messages

### CLI Usability
1. **Fast workflow**: `glassalpha audit` completes quickly
2. **Clear error messages**: Actionable feedback on failures
3. **Configuration validation**: Catch config errors early
4. **Progress indicators**: User feedback during operations

## Making Changes

### 1. Create a Feature Branch
```bash
git checkout -b feature/your-improvement
```

### 2. Focus on Audit Value
- **Ask**: Does this improve audit PDF quality or usability?
- **Test**: Does `make audit-test` still pass?
- **Document**: Update audit examples if changed

### 3. Test Your Changes
```bash
# Test suite
pytest --strict

# Run all tests
make test

# Code quality
black --check packages/src
ruff check packages/src
mypy --strict packages/src
```

### 4. Commit Your Changes
```bash
git add .
git commit -m "feat: improve PDF reproducibility

- Add config hash to manifest
- Fix seed handling in SHAP
- Closes #123"
```

## Pull Request Process

1. **Update documentation** for any new features
2. **Ensure all tests pass** locally
3. **Update the changelog** if applicable
4. **Submit PR** with clear description
5. **Address review feedback** promptly

### PR Title Format
- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation updates
- `test:` Test improvements

## Security & Privacy Guidelines

- **No PII in logs**: Never log personally identifiable information
- **No external calls**: Core library must work offline
- **Sanitize inputs**: Always validate and sanitize user inputs
- **Hash sensitive data**: Use SHA-256 for any ID hashing

## Documentation

### Docstring Format (Google Style)
```python
def explain_prediction(
    model: Any,
    instance: pd.Series,
    background: Optional[pd.DataFrame] = None
) -> Explanation:
    """Generate explanation for a single prediction.
    
    Args:
        model: Trained model object supporting .predict()
        instance: Single data instance to explain
        background: Background dataset for SHAP. If None, uses training data.
    
    Returns:
        Explanation object containing SHAP values and visualizations.
    
    Raises:
        ValueError: If instance shape doesn't match model input.
    
    Example:
        >>> exp = explain_prediction(model, X_test.iloc[0])
        >>> exp.plot_waterfall()
    """
```

### Examples
New features should include:
- Updated example notebooks as needed
- Before/after comparison if visual changes
- Performance impact measurement
- Updated tutorial if workflow changes

## Questions?

- 💬 [GitHub Discussions](https://github.com/GlassAlpha/glassalpha/discussions)
- 🐛 [Issue Tracker](https://github.com/GlassAlpha/glassalpha/issues)

Thank you for contributing to Glass Alpha! 🎉
