# Project Scope & Limitations

!!! warning "Pre-Alpha Development"
    GlassAlpha is under active development. This document describes our intended project scope and design limitations to ensure transparent expectations.

## Project Philosophy

GlassAlpha follows an **audit-first** approach, prioritizing regulatory compliance and trust over cutting-edge ML features.

### Current Development Scope

**What we're building:**
- Deterministic PDF audit report generation
- TreeSHAP explanations for tabular models (XGBoost, LightGBM, LogisticRegression)  
- Basic fairness metrics (demographic parity, equalized odds)
- Complete reproducibility with lineage tracking
- Simple CLI interface with YAML configuration

**What we're NOT building initially:**
- Deep learning or LLM support
- Real-time model serving
- Advanced bias mitigation algorithms
- Continuous monitoring systems
- Web dashboards or APIs
- Cloud-native features

## Intended Design Limitations

### Focus on Tabular Data

GlassAlpha is intentionally limited to tabular/structured data because:
- Most regulatory compliance requirements focus on tabular models
- TreeSHAP provides exact explanations for tree-based models
- Tabular fairness metrics are well-established
- Enterprise adoption is highest for tabular ML

### Determinism Over Performance

We prioritize reproducibility over speed:
- Every operation must be seedable and deterministic
- Byte-identical outputs are more important than fast execution
- All randomness must be explicitly controlled
- Performance optimizations cannot compromise reproducibility

### Local-First Architecture

GlassAlpha is designed for on-premise deployment:
- No cloud dependencies or external API calls
- All processing happens locally
- File-based storage instead of databases
- Privacy-preserving by default

## Target Scale & Performance

### Design Targets

GlassAlpha is being designed for:
- **Dataset size**: Up to 1M rows initially
- **Feature count**: Up to 200 features
- **Model complexity**: Up to 1000 trees
- **Report generation**: Under 60 seconds for standard audits

### Trade-offs

We explicitly choose:
- **Correctness over speed** - Accurate metrics matter more than fast execution
- **Completeness over scale** - Better to fully analyze smaller datasets
- **Clarity over complexity** - Simple, understandable metrics over advanced statistics

## Important Disclaimers

### Not Legal Advice

GlassAlpha provides technical tools for generating audit documentation. It does not:
- Provide legal advice or interpretations
- Guarantee regulatory compliance
- Replace human judgment in decision-making
- Certify models as compliant

**Always consult qualified legal counsel for compliance requirements.**

### Development Status

As a pre-alpha project:
- Features described in documentation may not be implemented
- APIs and configurations will change before v1.0
- Not suitable for production use
- No stability guarantees

## Responsible Use Guidelines

### Intended Use Cases

GlassAlpha is designed for:
- **Decision support** - Augmenting human decision-making
- **Compliance documentation** - Generating audit reports for regulators
- **Model validation** - Testing for bias and fairness issues
- **Development debugging** - Understanding model behavior

### Not Intended For

GlassAlpha should NOT be used for:
- **Fully automated decisions** without human review
- **Life-critical applications** (medical diagnosis, safety systems)
- **Real-time systems** requiring sub-second responses
- **Adversarial contexts** where explanations could be exploited

## Contributing & Feedback

Help us build GlassAlpha:
- **Code contributions**: See [Contributing Guide](contributing.md)
- **Bug reports**: [GitHub Issues](https://github.com/GlassAlpha/glassalpha/issues)
- **Feature requests**: [GitHub Discussions](https://github.com/GlassAlpha/glassalpha/discussions)
- **Documentation**: Help improve clarity and accuracy

## Project Evolution

GlassAlpha will continue to evolve based on community needs and contributions. Future improvements may focus on expanding compliance capabilities and supporting additional use cases as determined by user feedback and regulatory requirements.

---

*GlassAlpha: Building trust in machine learning, one audit at a time.*
