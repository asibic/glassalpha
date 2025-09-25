# Feature Comparison: OSS vs Enterprise

GlassAlpha offers a powerful open-source foundation with enterprise extensions for organizations requiring advanced capabilities, support, and compliance features.

## Feature Matrix

### Core Functionality

| Feature | OSS | Enterprise | Description |
|---------|:---:|:----------:|-------------|
| **Explainability** |
| TreeSHAP | ✅ | ✅ | Exact Shapley values for tree models |
| KernelSHAP | ✅ | ✅ | Model-agnostic explanations (slower) |
| DeepSHAP | ❌ | ✅ | Neural network explanations |
| GradientSHAP | ❌ | ✅ | Gradient-based explanations for LLMs |
| Attention Analysis | ❌ | ✅ | Transformer attention visualization |
| **Models Supported** |
| XGBoost | ✅ | ✅ | Full support with TreeSHAP |
| LightGBM | ✅ | ✅ | Full support with TreeSHAP |
| Logistic Regression | ✅ | ✅ | Basic ML model support |
| Random Forest | ✅ | ✅ | Tree ensemble support |
| Neural Networks | ❌ | ✅ | Deep learning models |
| LLMs (GPT, BERT, etc.) | ❌ | ✅ | Large language models |
| **Metrics & Analysis** |
| Performance Metrics | ✅ | ✅ | Accuracy, precision, recall, F1, AUC |
| Basic Fairness | ✅ | ✅ | Demographic parity, equal opportunity |
| Advanced Fairness | ❌ | ✅ | Conditional fairness, causal fairness |
| Drift Detection | ✅ | ✅ | PSI, KL divergence |
| Continuous Monitoring | ❌ | ✅ | Real-time drift tracking |

### Compliance & Reporting

| Feature | OSS | Enterprise | Description |
|---------|:---:|:----------:|-------------|
| **Report Generation** |
| Basic PDF Reports | ✅ | ✅ | Standard audit PDF |
| Custom Templates | ❌ | ✅ | Organization-specific branding |
| Interactive HTML | ❌ | ✅ | Web-based reports with drill-down |
| **Regulatory Templates** |
| Generic Audit | ✅ | ✅ | Basic compliance report |
| EU AI Act | ❌ | ✅ | EU-specific requirements |
| CFPB Compliance | ❌ | ✅ | US financial regulations |
| ISO/IEC 23053 | ❌ | ✅ | International AI standards |
| Custom Regulations | ❌ | ✅ | Industry-specific templates |
| **Recourse & Remediation** |
| Basic Recourse | ✅ | ✅ | Immutables, monotonicity |
| Advanced Recourse | ❌ | ✅ | Multi-objective optimization |
| Policy Packs | ❌ | ✅ | Pre-configured compliance rules |

### Infrastructure & Operations

| Feature | OSS | Enterprise | Description |
|---------|:---:|:----------:|-------------|
| **Deployment** |
| Local CLI | ✅ | ✅ | Command-line interface |
| API Server | ❌ | ✅ | REST/gRPC APIs |
| Container Support | ❌ | ✅ | Docker/Kubernetes ready |
| **Integrations** |
| File-based I/O | ✅ | ✅ | CSV, Parquet, JSON |
| AWS SageMaker | ❌ | ✅ | Native integration |
| Azure ML | ❌ | ✅ | Native integration |
| Databricks | ❌ | ✅ | Native integration |
| MLflow | ❌ | ✅ | Model registry integration |
| **Monitoring** |
| Static Analysis | ✅ | ✅ | One-time audits |
| Dashboard | ❌ | ✅ | Real-time monitoring UI |
| Alerting | ❌ | ✅ | Drift & fairness alerts |
| Audit Trail | ❌ | ✅ | Complete change history |

### Security & Governance

| Feature | OSS | Enterprise | Description |
|---------|:---:|:----------:|-------------|
| **Access Control** |
| Single User | ✅ | ✅ | Local execution |
| RBAC | ❌ | ✅ | Role-based access control |
| SSO/SAML | ❌ | ✅ | Enterprise authentication |
| Audit Logs | ❌ | ✅ | User activity tracking |
| **Data Privacy** |
| Local Processing | ✅ | ✅ | On-premise ready |
| Data Encryption | ❌ | ✅ | At-rest & in-transit |
| PII Redaction | ❌ | ✅ | Automatic PII handling |
| Differential Privacy | ❌ | ✅ | Privacy-preserving metrics |

### Support & Services

| Feature | OSS | Enterprise | Description |
|---------|:---:|:----------:|-------------|
| **Support Channels** |
| GitHub Issues | ✅ | ✅ | Community support |
| Email Support | ❌ | ✅ | Direct support channel |
| Priority Support | ❌ | ✅ | Guaranteed response times |
| Dedicated CSM | ❌ | ✅ | Customer success manager |
| **SLA** |
| Response Time | Best effort | 24 hours | Initial response |
| Resolution Time | Best effort | Guaranteed | Based on severity |
| Uptime Guarantee | N/A | 99.9% | For hosted services |
| **Training & Docs** |
| Public Documentation | ✅ | ✅ | Getting started guides |
| Advanced Guides | ❌ | ✅ | Best practices, architecture |
| Custom Training | ❌ | ✅ | Organization-specific |
| Certification | ❌ | ✅ | Professional certification |

## Licensing

### OSS License (Apache 2.0)
- ✅ Free for any use (commercial or non-commercial)
- ✅ Modify and distribute freely
- ✅ No warranty or liability
- ✅ Community-driven development

### Enterprise License
- 💰 Annual subscription model
- 🏢 Pricing based on organization size and usage
- 🔒 Includes indemnification and warranties
- 📞 Direct vendor support
- 🚀 Priority feature requests

## Getting Started

### OSS Installation
```bash
git clone https://github.com/GlassAlpha/glassalpha
cd glassalpha/packages
pip install -e .
glassalpha audit --config configs/german_credit_simple.yaml --output report.pdf
```

### Enterprise Installation
```bash
# Contact enterprise team for license key and installation
export GLASSALPHA_LICENSE_KEY="your-license-key"
pip install glassalpha-enterprise
glassalpha audit --config audit.yaml --output report.pdf --strict
```

## Contact

- **OSS Support**: [GitHub Issues](https://github.com/GlassAlpha/glassalpha/issues)
- **Enterprise Sales**: sales@glassalpha.ai
- **Enterprise Support**: support@glassalpha.ai (customers only)

## Current Capabilities

### OSS Features (Available Now)
- ✅ Tabular model audits (XGBoost, LightGBM, Logistic Regression)
- ✅ TreeSHAP and KernelSHAP explanations
- ✅ Comprehensive fairness and performance metrics
- ✅ Professional PDF report generation
- ✅ Complete audit trail and reproducibility
- ✅ CLI interface with YAML configuration

### Enterprise Features (Contact for Access)
- 🏢 Advanced explainability methods
- 🏢 Custom regulatory templates
- 🏢 Priority support and SLAs
- 🏢 Advanced integrations and connectors
- 🏢 Continuous monitoring capabilities
- 🏢 Role-based access control and audit logging

---

*Last updated: September 2024*
