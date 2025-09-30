# GlassAlpha

**Open-source AI compliance toolkit for transparent, auditable, and regulator-ready ML models.**

## Why GlassAlpha?

As AI regulations tighten globally (EU AI Act, CFPB guidance), organizations need **transparent, auditable ML systems**. Most existing audit tools are either academic research code, enterprise SaaS platforms with vendor lock-in, or custom internal tools that lack reproducibility.

GlassAlpha provides **deterministic, regulator-ready audit reports** with complete lineage tracking. Run the same config twice, get byte-identical PDFs. Every decision is explainable, every metric is reproducible, every audit trail is complete.

**Why Open Source**: Compliance tools require trust. Open source code means regulators, auditors, and your team can verify exactly how conclusions were reached.

## Quick Start

### Installation & First Audit

```bash
# Clone and setup
git clone https://github.com/GlassAlpha/glassalpha
cd glassalpha/packages

# Install and verify
pip install -e .
glassalpha --help

# Generate your first audit PDF (under 60 seconds)
glassalpha audit --config configs/german_credit_simple.yaml --output audit.pdf
```

**[→ See detailed installation and setup guide](packages/README.md#installation)**

## Project Structure

- **`packages/`**: Core Python package ([developer documentation](packages/README.md))
- **`site/`**: User documentation (MkDocs)
- **`configs/`**: Example audit configurations

## Key Features

**Production Ready**:

- ✅ One-command PDF audit generation
- ✅ XGBoost, LightGBM, LogisticRegression support
- ✅ TreeSHAP explainability + fairness metrics
- ✅ Deterministic, reproducible outputs
- ✅ Apache 2.0 license with optional Enterprise extensions

## Documentation

- **[Full Documentation](https://glassalpha.com/)** - Complete user guides and tutorials
- **[Developer Guide](packages/README.md)** - Architecture, development, and contribution details
- **[German Credit Tutorial](https://glassalpha.com/examples/german-credit-audit/)** - Step-by-step example

## Contributing

We welcome contributions! See [Contributing Guidelines](https://glassalpha.com/reference/contributing/) for development setup and guidelines.

## License

- **Core Library**: Apache 2.0 License - see [LICENSE](LICENSE)
- **Enterprise Extensions**: Commercial license for advanced features
- **Brand**: "GlassAlpha" trademark - see [TRADEMARK.md](TRADEMARK.md)

**[→ See detailed licensing information](packages/README.md#license--dependencies)**

---

_Building trust through transparency in AI compliance._
