# GlassAlpha

**Open-source AI compliance toolkit for transparent, auditable, and regulator-ready ML models.**

## Why GlassAlpha?

As AI regulations tighten globally (EU AI Act, CFPB guidance), organizations need **transparent, auditable ML systems**. Most existing audit tools are either academic research code, enterprise SaaS platforms with vendor lock-in, or custom internal tools that lack reproducibility.

GlassAlpha provides **deterministic, regulator-ready audit reports** with complete lineage tracking. Run the same config twice, get byte-identical PDFs. Every decision is explainable, every metric is reproducible, every audit trail is complete.

**Why Open Source**: Compliance tools require trust. Open source code means regulators, auditors, and your team can verify exactly how conclusions were reached.

## Status & Scope

**Production Features**:
- ✅ One-command PDF audit generation
- ✅ XGBoost, LightGBM, LogisticRegression support
- ✅ TreeSHAP explainability + fairness metrics
- ✅ Deterministic, reproducible outputs
- ✅ Apache 2.0 license with optional Enterprise extensions

## Quick Start

### Installation

```bash
# Clone and setup
git clone https://github.com/GlassAlpha/glassalpha
cd glassalpha/packages

# Install
pip install -e .

# Verify installation
glassalpha --help
```

### Generate Your First Audit
```bash
# Generate audit PDF
glassalpha audit --config configs/german_credit_simple.yaml --output audit.pdf
```

## Project Structure

- **`packages/`**: Core Python package ([detailed README](packages/README.md))
  - Plugin architecture with OSS/Enterprise separation
  - Protocol-based interfaces for extensibility
  - Deterministic component selection
- **`site/`**: Documentation (MkDocs)
- **`configs/`**: Example audit configurations

## Architecture Highlights

GlassAlpha is built as an **extensible framework**:
- **Plugin Architecture**: Models, explainers, and metrics use dynamic registration
- **OSS/Enterprise Split**: Core functionality open-source, advanced features commercial
- **Deterministic Design**: Configuration-driven with reproducible results
- **Audit Profiles**: Different compliance contexts use appropriate component sets

For detailed architecture information, see the [Architecture Guide](site/docs/architecture.md).

👉 **See [packages/README.md](packages/README.md) for detailed architecture, installation, and usage.**

## Contributing

We welcome contributions! Areas for enhancement:
- Additional model support
- Advanced explainability methods
- Extended compliance frameworks
- Performance optimizations

See [CONTRIBUTING](site/docs/contributing.md) for development guidelines.

## Documentation

- **Quick Reference**: [packages/README.md](packages/README.md)
- **Full Docs**: [site/docs/](site/docs/)
- **Examples**: [German Credit Tutorial](site/docs/examples/german-credit-audit.md)

## License & Business Model

- **Core Library**: Apache 2.0 License ([LICENSE](LICENSE))
- **Enterprise Extensions**: Commercial license for advanced features
- **Brand**: "GlassAlpha" trademark - see [TRADEMARK.md](TRADEMARK.md)

**Support**:
- OSS: GitHub Issues
- Enterprise: Priority support with SLA

---

*Building trust through transparency in AI compliance.*
