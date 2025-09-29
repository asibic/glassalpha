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

# Python 3.11 or 3.12 recommended
python3 --version   # should show 3.11.x or 3.12.x

# (Recommended) Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install in editable mode
python -m pip install --upgrade pip
pip install -e ".[dev]"

# Verify installation
glassalpha --help
```

### Generate Your First Audit

Generate audit PDF:

```bash
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

## License & Dependencies

GlassAlpha provides **enterprise-grade licensing compatibility** with a carefully curated technology stack. All dependencies are chosen for regulatory compliance, commercial viability, and compatibility with the Apache 2.0 license.

### Core License Structure

- **GlassAlpha Framework**: Apache License 2.0 ([LICENSE](LICENSE))
- **Enterprise Extensions**: Commercial license for potential future advanced features
- **Brand**: "GlassAlpha" trademark - see [TRADEMARK.md](TRADEMARK.md)

### Technology Stack & Licenses

GlassAlpha uses industry-standard, enterprise-compatible dependencies with proven licensing compatibility:

| Component        | License     | Purpose                     | Why Chosen                                     |
| ---------------- | ----------- | --------------------------- | ---------------------------------------------- |
| **Python SHAP**  | MIT License | TreeSHAP explanations       | ✅ Enterprise-compatible, no GPL contamination |
| **XGBoost**      | Apache 2.0  | Gradient boosting models    | ✅ Same license family, proven in production   |
| **LightGBM**     | MIT License | Alternative tree models     | ✅ Microsoft-backed, widely adopted            |
| **scikit-learn** | BSD License | Baseline models & utilities | ✅ Academic standard, fully compatible         |
| **NumPy**        | BSD License | Numerical computing         | ✅ Core scientific Python library              |
| **Pandas**       | BSD License | Data manipulation           | ✅ Industry standard for data science          |
| **WeasyPrint**   | BSD License | PDF generation              | ✅ Pure Python, no system dependencies         |
| **Typer**        | MIT License | CLI framework               | ✅ Modern, type-safe command interface         |
| **Pydantic**     | MIT License | Configuration validation    | ✅ Runtime type checking and validation        |

### Licensing Confidence & Risk Mitigation

**✅ No GPL Dependencies**: GlassAlpha deliberately avoids GPL-licensed components to ensure maximum compatibility with enterprise environments. We use the MIT-licensed Python [SHAP](https://github.com/shap/shap) library rather than the GPL-licensed R `treeshap` package.

**✅ Apache 2.0 Compatible Stack**: All dependencies are compatible with Apache 2.0 licensing, allowing:

- Commercial use without restrictions
- Integration with proprietary systems
- Distribution in closed-source applications
- Patent protection for contributors

**✅ Regulatory Compliance Ready**: The licensing structure supports:

- Audit trail preservation
- Reproducible builds with locked dependency versions
- No vendor lock-in through open standards
- Full source code transparency for regulatory review

### Enterprise Integration

The clean licensing structure enables:

- **Container Integration**: Deploy in Docker/Kubernetes without license conflicts
- **CI/CD Pipelines**: Automated builds with reproducible dependency resolution
- **Cloud Deployment**: Compatible with AWS, Azure, GCP licensing requirements
- **On-Premise Installation**: Full control over software stack and dependencies

### Dependency Verification

All dependencies are locked to specific versions in [`constraints.txt`](packages/constraints.txt) for reproducible builds:

```bash
# Reproducible installation
pip install -c constraints.txt -e .
```

**Support**:

- OSS: GitHub Issues

---

_Building trust through transparency in AI compliance._
