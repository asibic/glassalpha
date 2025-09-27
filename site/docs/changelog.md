# Changelog

All notable changes to GlassAlpha are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- **Professional Presentation**: Production-ready documentation suitable for compliance contexts

### Architecture
- **Plugin System**: Registry-based architecture for models, explainers, and metrics
- **Enterprise Ready**: Feature flag system and clear OSS/Enterprise boundaries
- **Extension Points**: Protocol-based interfaces enabling custom implementations
- **Audit Profiles**: Component configuration sets for different compliance requirements

## Release History

### [0.1.0] - September 24, 2024

**Production Release - Core ML Auditing Capabilities**

This release provides complete functionality for professional ML model auditing with:

- Working CLI that generates PDF reports in under 60 seconds
- 5 model wrappers, 3 explainer implementations, 17 metrics
- Complete German Credit audit example with regulatory analysis
- Deterministic, reproducible results suitable for compliance review

**Verification Commands:**
```bash
glassalpha --version                    # v0.1.0
glassalpha list                         # Show all available components
glassalpha audit --config configs/german_credit_simple.yaml --output audit.pdf
```

**System Requirements:**
- Python 3.11+
- 2GB RAM minimum (8GB recommended)
- macOS 10.15+, Linux (Ubuntu 20.04+), Windows 10+ (WSL2 recommended)

See [Installation Guide](getting-started/installation.md) for complete setup instructions.

## Development Timeline

- **September 2024**: Architecture foundation and core component implementation
- **September 2024**: Integration pipeline and report generation system
- **September 2024**: End-to-end testing and documentation completion
- **September 24, 2024**: Production release v0.1.0

## Potential Future Enhancement Areas

Potential future releases may include enhancements based on community needs:

- Additional model type support
- Extended compliance framework coverage
- Enhanced integration capabilities
- Advanced explanation methods

Enhancement priorities are determined by user feedback and enterprise customer requirements.

---

For detailed release notes and downloads, visit [GitHub Releases](https://github.com/GlassAlpha/glassalpha/releases).
