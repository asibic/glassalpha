# Project Scope & Limitations

GlassAlpha focuses on professional ML audit capabilities for regulated industries. This document outlines our design philosophy, intended scope, and responsible use guidelines.

## Design Philosophy

GlassAlpha follows an **audit-first** approach, prioritizing regulatory compliance and trust over cutting-edge ML features.

### Core Capabilities

**What GlassAlpha provides:**
- Deterministic PDF audit report generation
- TreeSHAP explanations for tabular models (XGBoost, LightGBM, Logistic Regression)
- Comprehensive fairness metrics (demographic parity, equal opportunity, etc.)
- Complete reproducibility with audit trail tracking
- Professional CLI interface with YAML configuration
- Regulatory framework compliance support

### Intentional Scope Limitations

**Current focus areas:**
- Tabular/structured data analysis
- Classification model auditing
- On-premise deployment
- File-based workflows
- Deterministic operations

**Not currently supported:**
- Deep learning model analysis (neural networks, LLMs)
- Real-time model monitoring
- Cloud-native deployment features
- Web-based interfaces
- Automated bias mitigation

## Design Trade-offs

### Focus on Tabular Data

GlassAlpha is intentionally designed for tabular/structured data because:

- Most regulated ML applications use tabular models
- TreeSHAP provides exact explanations for tree-based models
- Tabular fairness metrics have regulatory precedent
- Enterprise adoption is highest for tabular ML in regulated industries

### Determinism Over Performance

We prioritize reproducibility over speed:

- Every operation is seedable and deterministic
- Byte-identical outputs ensure audit integrity
- All randomness is explicitly controlled
- Performance optimizations cannot compromise reproducibility

### Local-First Architecture

GlassAlpha is designed for secure, on-premise deployment:

- No external API calls or cloud dependencies
- All processing happens locally
- File-based storage ensures data sovereignty
- Privacy-preserving by default

## Scale & Performance Characteristics

### Design Targets

GlassAlpha is optimized for:
- **Dataset size**: Up to 1M rows (tested extensively)
- **Feature count**: Up to 1,000 features
- **Model complexity**: Up to 1,000 trees
- **Report generation**: Under 60 seconds for standard audits

### Performance Profile

Typical execution times:
- **Small datasets** (<1K rows): 1-3 seconds
- **Medium datasets** (1-10K rows): 3-15 seconds
- **Large datasets** (10-100K rows): 15-60 seconds
- **Very large datasets** (100K+ rows): 1-5 minutes

### Optimization Guidelines

For optimal performance:
- Use TreeSHAP with tree-based models (XGBoost, LightGBM)
- Reduce explainer sample sizes for large datasets
- Enable parallel processing on multi-core systems
- Use SSD storage for better I/O performance

## Important Disclaimers

### Professional Tool, Not Legal Advice

GlassAlpha provides technical audit capabilities. It does not:
- Provide legal advice or compliance interpretations
- Guarantee regulatory approval or compliance
- Replace qualified legal counsel review
- Certify models as discrimination-free

**Always consult qualified legal and compliance professionals for regulatory requirements.**

### Model Limitations

Users should understand:
- **Bias detection** identifies statistical disparities, not causal discrimination
- **Explanations** show correlations, not causal relationships
- **Fairness metrics** reflect statistical parity, not equitable outcomes
- **Audit reports** document technical findings, not legal conclusions

## Responsible Use Guidelines

### Appropriate Use Cases

GlassAlpha is designed for:
- **Compliance documentation** - Generating audit reports for regulatory review
- **Model validation** - Testing for bias and performance issues
- **Decision support** - Augmenting human decision-making processes
- **Development workflows** - Understanding model behavior during development

### Important Limitations

GlassAlpha should NOT be used as:
- **Sole basis for deployment decisions** without human review
- **Substitute for human judgment** in high-stakes decisions
- **Real-time system component** requiring sub-second responses
- **Adversarial defense** where explanations might be exploited

### Risk Considerations

Be aware of:
- **Model limitations** affecting fairness and accuracy
- **Data quality issues** that impact audit validity
- **Explanation limitations** in complex or adversarial scenarios
- **Regulatory interpretation** requiring legal expertise

## Technical Limitations

### Model Support

**Fully Supported:**
- XGBoost (with native TreeSHAP)
- LightGBM (with native TreeSHAP)
- Logistic Regression (with KernelSHAP)
- Generic scikit-learn classifiers (with KernelSHAP)

**Not Currently Supported:**
- Neural networks and deep learning models
- Large language models (LLMs)
- Computer vision models
- Time series models
- Multi-output or structured prediction models

### Explanation Methods

**Available:**
- **TreeSHAP** - Exact Shapley values for tree models (fast, accurate)
- **KernelSHAP** - Model-agnostic explanations (slower, approximate)

**Limitations:**
- TreeSHAP only works with tree-based models
- KernelSHAP can be slow for complex models
- Explanations show feature importance, not causal impact
- Local explanations may not represent global model behavior

### Data Requirements

**Supported Formats:**
- CSV, Parquet, Feather, Pickle
- Tabular data with mixed data types
- Missing values (with preprocessing options)

**Requirements:**
- Well-defined target variable for supervised learning
- Protected attributes identifiable in data
- Sufficient sample sizes for statistical testing
- Reasonable feature-to-sample ratios

## Potential Future Enhancement Areas

GlassAlpha may evolve to include:
- Enhanced model type support based on user needs
- Additional explanation methods for specialized use cases
- Extended compliance framework coverage
- Integration capabilities with enterprise workflows

These enhancements depend on community contributions and enterprise customer requirements.

## Support and Community

### Getting Help

- **Documentation**: Comprehensive guides and examples
- **Community**: GitHub Discussions for user questions
- **Issues**: GitHub Issues for bug reports
- **Enterprise**: Dedicated support for commercial users

### Contributing

Help improve GlassAlpha:
- **Code contributions**: See [Contributing Guide](contributing.md)
- **Bug reports**: [GitHub Issues](https://github.com/GlassAlpha/glassalpha/issues)
- **Feature suggestions**: [GitHub Discussions](https://github.com/GlassAlpha/glassalpha/discussions)
- **Documentation**: Help improve clarity and coverage

## Professional Standards

GlassAlpha maintains professional standards appropriate for regulated industries:

- **Quality assurance** through comprehensive testing
- **Documentation standards** for regulatory review
- **Reproducibility guarantees** for audit integrity
- **Security practices** for sensitive data handling
- **Compliance focus** aligned with regulatory requirements

This ensures GlassAlpha serves as a reliable foundation for ML governance and compliance in professional environments.

---

*GlassAlpha: Professional ML auditing for regulated industries.*
