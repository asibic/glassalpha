# About GlassAlpha

GlassAlpha is built as an **extensible framework** for AI compliance and interpretability, starting with tabular ML models in Phase 1 and designed to expand to LLMs and multimodal systems.

## Why GlassAlpha?

### The Problem

As AI regulations tighten globally (EU AI Act, CFPB guidance, state-level bills), regulated industries are most exposed. Finance, insurance, healthcare, and defense increasingly rely on ML models for high-stakes decisions: loan approvals, fraud detection, underwriting, patient triage, targeting systems, etc.

These organizations need **transparent, auditable ML systems**. But most ML models are black boxes, and existing audit tools are either:

- Academic research code (not production-ready)
- Enterprise SaaS platforms (vendor lock-in, data privacy concerns)
- Custom internal tools (inconsistent, non-reproducible)

### The Solution

GlassAlpha provides **deterministic, regulator-ready audit reports** with complete lineage tracking. Run the same config twice, get byte-identical PDFs. Every decision is explainable, every metric is reproducible, every audit trail is complete.

### Who This Helps

- **Data Scientists**: Generate compliance documentation without manual report writing
- **Legal/Compliance Teams**: Get standardized, defensible audit reports for regulatory review
- **Risk Managers**: Monitor model fairness and drift with quantitative metrics
- **Auditors**: Verify model behavior with reproducible, transparent analysis

### Why Open Source

Compliance tools require trust. Open source code means regulators, auditors, and your team can verify exactly how conclusions were reached. No proprietary black boxes auditing your black boxes.

## Technical Foundation

GlassAlpha is built on a plugin architecture designed for extensibility and regulatory compliance. For detailed technical information, see the [Architecture Guide](reference/architecture.md).

## Getting Started

Ready to try GlassAlpha? Start with the [Quick Start Guide](getting-started/quickstart.md) or try the [5-minute tutorial](examples/quick-start-audit.md) to generate your first audit.
