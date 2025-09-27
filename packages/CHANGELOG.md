# Changelog

All notable changes to GlassAlpha are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### In Development
- Ongoing improvements to test coverage and code quality
- Enhanced error handling and validation
- Performance optimizations for large datasets

## [0.1.0] - 2024-09-24

### Added
- **Core ML Model Support**: XGBoost, LightGBM, Logistic Regression, and generic scikit-learn classifiers
- **Explanation Methods**: TreeSHAP for tree models, KernelSHAP for model-agnostic explanations
- **Comprehensive Metrics**: 17 metrics including performance, fairness, and drift detection
- **Professional Reporting**: PDF generation with deterministic plots and audit trails
- **CLI Interface**: Complete command-line tool with `audit`, `validate`, and `list` commands
- **Configuration System**: YAML-based configuration with Pydantic validation and strict mode
- **Data Processing**: Tabular data loader with schema validation and protected attribute handling
- **Reproducibility**: Deterministic execution with seed management and audit manifests
- **German Credit Dataset**: Complete working example with regulatory interpretation

### Documentation
- **Complete User Guides**: Installation, quickstart, configuration, and CLI reference
- **API Reference**: Comprehensive documentation for all public interfaces
- **Compliance Framework**: GDPR, ECOA, FCRA regulatory mapping and guidance
- **Troubleshooting Guide**: Common issues and solutions
- **FAQ Section**: Comprehensive answers to user questions

### Architecture
- **Plugin System**: Registry-based architecture for models, explainers, and metrics
- **Enterprise Ready**: Feature flag system and clear OSS/Enterprise boundaries
- **Extension Points**: Protocol-based interfaces enabling custom implementations
- **Audit Profiles**: Component configuration sets for different compliance requirements

### Technical Features
- **Deterministic Operations**: All randomness controlled via seed management system
- **Audit Manifests**: Complete provenance tracking with Git SHA, data hashes, and component versions
- **Strict Mode**: Regulatory compliance enforcement with explicit validation
- **Performance Optimizations**: Multi-core processing and memory-efficient operations
- **Security**: Local-only processing with no external network dependencies

### Supported Models
- **XGBoost**: Full TreeSHAP integration with native performance optimizations
- **LightGBM**: Complete TreeSHAP support with Microsoft's gradient boosting
- **Logistic Regression**: Scikit-learn integration with KernelSHAP explanations
- **Generic Scikit-learn**: Support for most scikit-learn classifiers

### Metrics Coverage
- **Performance (6)**: Accuracy, Precision, Recall, F1, AUC-ROC, Classification Report
- **Fairness (4)**: Demographic Parity, Equal Opportunity, Equalized Odds, Predictive Parity
- **Drift (7)**: PSI, KL Divergence, KS Test, JS Divergence, Prediction Drift, Feature Drift, Target Drift

### Quality Assurance
- **Test Coverage**: Comprehensive test suite with >50% coverage
- **Type Safety**: Full mypy strict mode compliance
- **Code Quality**: Automated linting with ruff and black formatting
- **CI/CD Pipeline**: Automated testing across Python 3.11 and 3.12

## Release History

### [0.1.0] - September 24, 2024

**Production Release - Core ML Auditing Capabilities**

This release provides complete functionality for professional ML model auditing:

- ✅ Working CLI that generates PDF reports in under 60 seconds
- ✅ 5 model wrappers, 3 explainer implementations, 17 metrics
- ✅ Complete German Credit audit example with regulatory analysis
- ✅ Deterministic, reproducible results suitable for compliance review
- ✅ Professional documentation suitable for enterprise adoption

**Quick Verification:**
```bash
pip install -e .
glassalpha --version                    # Should show v0.1.0
glassalpha list                         # Show all available components
glassalpha audit --config configs/german_credit_simple.yaml --output audit.pdf
```

**System Requirements:**
- Python 3.11 or higher
- 2GB RAM minimum (8GB recommended for large datasets)
- macOS 10.15+, Linux (Ubuntu 20.04+), Windows 10+ (WSL2 recommended)
- 1GB disk space for installation and temporary files

## Development Milestones

- **Phase 0 (Architecture)**: Foundation interfaces, registry system, and component patterns
- **Phase 1 (Core Features)**: Model integration, explanation generation, metrics computation
- **Phase 1 (Integration)**: End-to-end pipeline, report generation, and CLI interface
- **Phase 1 (Quality)**: Testing, documentation, and production readiness
- **v0.1.0 Release**: Production-ready core ML auditing capabilities

## Breaking Changes

### From Pre-Release to 0.1.0
- Configuration schema solidified - old config formats not supported
- CLI interface finalized - command syntax is now stable
- API interfaces locked - protocol definitions are now stable

## Migration Guide

### To 0.1.0
This is the first production release. For users upgrading from development versions:

1. **Update Configuration**: Use new YAML schema with `audit_profile` specification
2. **CLI Changes**: Use `glassalpha audit` instead of development commands
3. **Dependencies**: Install from requirements with `pip install -e .`

## Known Issues

### Version 0.1.0
- **Large Datasets**: Performance may degrade for datasets >100K rows (optimization in progress)
- **Memory Usage**: KernelSHAP can be memory-intensive for complex models
- **Windows**: Native Windows support exists but WSL2 recommended for best experience

## Potential Future Enhancement Areas

Future releases may include enhancements based on community needs and enterprise requirements:

- **Model Support**: Additional ML libraries and model types
- **Compliance**: Extended regulatory framework coverage
- **Performance**: Large-scale processing and optimization
- **Integration**: Enterprise system connectors and APIs
- **Explanation**: Advanced interpretability methods

Enhancement priorities are determined by user feedback, community contributions, and enterprise customer requirements.

---

For detailed release notes, installation instructions, and usage examples, see the [full documentation](https://glassalpha.com).

For issues and feature requests, visit [GitHub Issues](https://github.com/GlassAlpha/glassalpha/issues).
