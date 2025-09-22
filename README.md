# GlassAlpha

**Open-source AI compliance toolkit for transparent, auditable, and regulator-ready tabular ML models.**

> ⚠️ **Pre-Alpha Status**: GlassAlpha is under active development. Core functionality is being built to deliver deterministic, regulator-ready PDF audits.

## Vision

GlassAlpha provides **one-command audit generation** that produces deterministic, professional PDF reports for ML models - designed for teams needing reproducible, audit-ready documentation.

### Phase 1 Focus (Current)
- **PDF Audit Generation**: Polished, exportable reports with complete lineage tracking
- **Deterministic Output**: Byte-identical PDFs with same seed/data/model
- **TreeSHAP Explainability**: Feature importance and individual predictions
- **Basic Recourse**: Immutables, monotonic constraints, and cost functions  
- **Supported Models**: XGBoost, LightGBM, Logistic Regression

## Quick Start

### Development Setup

```bash
# Clone and enter repository
git clone https://github.com/GlassAlpha/glassalpha
cd glassalpha

# Set up Python environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package with dev dependencies
pip install -e packages/[dev]

# Run tests
pytest

# (Optional) View documentation locally
cd site && pip install -r requirements.txt && mkdocs serve
```

### Target Usage (Coming Soon)

```bash
# Generate deterministic audit PDF
glassalpha audit --config configs/audit.yaml --out audit.pdf
```

## Project Structure

- `packages/`: Core Python package (`glassalpha`)
- `site/`: Documentation (MkDocs)
- `configs/`: Policy and audit configurations (planned)
- `examples/`: Example notebooks and audits (planned)

## Key Design Principles

- **On-premise first**: No external dependencies or network calls
- **Reproducible**: Immutable run manifests with hashes and seeds
- **Config-driven**: YAML-based policy-as-code
- **Fast execution**: Under 60 seconds from model to PDF (target)

## Documentation

Full documentation available at [site/docs/](site/docs/) or view online (once deployed).

## Contributing

We welcome contributions! Priority areas:
- Core audit engine and PDF generation
- TreeSHAP integration
- CLI interface  
- Test coverage with German Credit and Adult Income datasets

See [CONTRIBUTING](site/docs/contributing.md) for guidelines.

## License & Trademark

**Code License**: Apache-2.0 - See [LICENSE](LICENSE)

**Trademark Notice**: "GlassAlpha", "Glass Alpha" and the GlassAlpha logo are trademarks protected under trademark law. See [TRADEMARK.md](TRADEMARK.md) for usage guidelines. While our code is open source, our brand is not.
