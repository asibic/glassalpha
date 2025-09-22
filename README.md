# Glass Alpha

**Open-source AI compliance toolkit for transparent, auditable, and regulator-ready ML models.**

> ⚠️ **Pre-Alpha Status**: GlassAlpha is under active development. Core functionality is being built to deliver deterministic, regulator-ready PDF audits.

## Why Glass Alpha?

As AI regulations tighten globally (EU AI Act, CFPB guidance), organizations need **transparent, auditable ML systems**. Most existing audit tools are either academic research code, enterprise SaaS platforms with vendor lock-in, or custom internal tools that lack reproducibility.

Glass Alpha provides **deterministic, regulator-ready audit reports** with complete lineage tracking. Run the same config twice, get byte-identical PDFs. Every decision is explainable, every metric is reproducible, every audit trail is complete.

**Why Open Source**: Compliance tools require trust. Open source code means regulators, auditors, and your team can verify exactly how conclusions were reached.

## Status & Scope

**Phase 1 Focus** (Current Development):
- ✅ One-command PDF audit generation  
- ✅ XGBoost, LightGBM, LogisticRegression support
- ✅ TreeSHAP explainability + fairness metrics
- ✅ Deterministic, reproducible outputs
- ✅ Apache 2.0 license with optional Enterprise extensions

## Quick Start

> ⚠️ **Pre-Alpha Status**: GlassAlpha is under active development. Core functionality is being built to deliver deterministic, regulator-ready PDF audits.

### Development Setup

```bash
# Clone and setup
git clone https://github.com/GlassAlpha/glassalpha
cd glassalpha

# Install with dev dependencies
cd packages && pip install -e ".[dev]"

# Run tests
pytest
```

### Target Usage
```bash
# Generate audit PDF (coming soon)
glassalpha audit --config configs/audit.yaml --out audit.pdf --strict
```

## Project Structure

- **`packages/`**: Core Python package ([detailed README](packages/README.md))
  - Plugin architecture with OSS/Enterprise separation
  - Protocol-based interfaces for extensibility  
  - Deterministic component selection
- **`site/`**: Documentation (MkDocs)
- **`configs/`**: Example audit configurations

## Architecture Highlights

Glass Alpha is built as an **extensible framework**:
- **Plugin Architecture**: Models, explainers, and metrics use dynamic registration
- **OSS/Enterprise Split**: Core functionality open-source, advanced features commercial
- **Deterministic Behavior**: Configuration-driven with reproducible results
- **Audit Profiles**: Different compliance contexts use appropriate component sets

👉 **See [packages/README.md](packages/README.md) for detailed architecture, installation, and usage.**

## Contributing

We welcome contributions! Current priorities:
- Core audit engine and PDF generation
- TreeSHAP integration and fairness metrics
- CLI interface with strict mode
- Test coverage with standard datasets

See [CONTRIBUTING](site/docs/contributing.md) for development guidelines.

## Documentation

- **Quick Reference**: [packages/README.md](packages/README.md)
- **Full Docs**: [site/docs/](site/docs/) 
- **Examples**: Coming soon in Phase 1

## License & Business Model

- **Core Library**: Apache 2.0 License ([LICENSE](LICENSE))
- **Enterprise Extensions**: Commercial license for advanced features
- **Brand**: "Glass Alpha" trademark - see [TRADEMARK.md](TRADEMARK.md)

**Support**:
- OSS: GitHub Issues  
- Enterprise: Priority support with SLA

---

*Building trust through transparency in AI compliance.*