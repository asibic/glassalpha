# Licensing & Dependencies

GlassAlpha provides **enterprise-grade licensing compatibility** with a carefully curated technology stack. This page provides detailed information about licensing, dependencies, and compliance considerations.

## Core License Structure

### GlassAlpha Framework License

- **License**: Apache License 2.0
- **File**: [LICENSE](https://www.apache.org/licenses/LICENSE-2.0.txt)
- **Scope**: Core framework, CLI tools, and OSS features

### Brand & Trademark

- **Trademark**: "GlassAlpha" name and logo
- **Policy**: See [TRADEMARK](../TRADEMARK.md)
- **Usage**: Requires explicit permission except as described in trademark policy

## Technology Stack & Licenses

GlassAlpha uses industry-standard, enterprise-compatible dependencies with proven licensing compatibility:

| Component        | Version | License            | Purpose                     | Why Chosen                                     |
| ---------------- | ------- | ------------------ | --------------------------- | ---------------------------------------------- |
| **Python SHAP**  | 0.44.1  | MIT License        | TreeSHAP explanations       | ✅ Enterprise-compatible, no GPL contamination |
| **XGBoost**      | 2.1.1   | Apache 2.0         | Gradient boosting models    | ✅ Same license family, proven in production   |
| **LightGBM**     | 4.1.0   | MIT License        | Alternative tree models     | ✅ Microsoft-backed, widely adopted            |
| **scikit-learn** | 1.3.2   | BSD License        | Baseline models & utilities | ✅ Academic standard, fully compatible         |
| **NumPy**        | 1.26.4  | BSD License        | Numerical computing         | ✅ Core scientific Python library              |
| **Pandas**       | 2.2.2   | BSD License        | Data manipulation           | ✅ Industry standard for data science          |
| **SciPy**        | 1.11.4  | BSD License        | Scientific computing        | ✅ Essential scientific Python library         |
| **WeasyPrint**   | 61.2    | BSD License        | PDF generation              | ✅ Pure Python, no system dependencies         |
| **Matplotlib**   | 3.7+    | Matplotlib License | Plotting for reports        | ✅ Standard Python plotting library            |
| **Seaborn**      | 0.12+   | BSD License        | Statistical plotting        | ✅ Built on matplotlib, enhanced stats         |
| **Typer**        | 0.12.3  | MIT License        | CLI framework               | ✅ Modern, type-safe command interface         |
| **Pydantic**     | 2.8.2   | MIT License        | Configuration validation    | ✅ Runtime type checking and validation        |
| **PyYAML**       | 6.0.1   | MIT License        | YAML config parsing         | ✅ Standard YAML processing                    |
| **orjson**       | 3.9+    | Apache 2.0 / MIT   | Fast JSON serialization     | ✅ High-performance JSON handling              |

_See [constraints.txt](../constraints.txt) for exact locked versions used in CI._

## Licensing Confidence & Risk Mitigation

### No GPL Dependencies

**✅ Clean License Stack**: GlassAlpha deliberately avoids GPL-licensed components to ensure maximum compatibility with enterprise environments.

**Critical Distinction**: We use the MIT-licensed Python [SHAP](https://github.com/shap/shap) library rather than the GPL-licensed R `treeshap` package for TreeSHAP explanations.

### Apache 2.0 Compatible Stack

All dependencies are compatible with Apache 2.0 licensing, enabling:

- **Commercial Use**: No restrictions on commercial applications
- **Proprietary Integration**: Can be embedded in closed-source systems
- **Distribution Freedom**: No copyleft restrictions on distribution
- **Patent Protection**: Includes patent grants for contributors

### Regulatory Compliance Ready

The licensing structure supports regulatory requirements:

- **Audit Trail Preservation**: Full source code transparency for regulatory review
- **Reproducible Builds**: Locked dependency versions ensure consistent results
- **No Vendor Lock-in**: Open standards and interfaces prevent dependency lock-in
- **Compliance Documentation**: Complete license documentation for audit trails

## Enterprise Integration Compatibility

### Container & Cloud Deployment

The clean licensing structure enables seamless integration:

- **Docker/Kubernetes**: Deploy without license conflicts
- **CI/CD Pipelines**: Automated builds with reproducible dependency resolution
- **Cloud Platforms**: Compatible with AWS, Azure, GCP licensing requirements
- **On-Premise**: Full control over software stack and dependencies

### Legal Compliance

- **Export Controls**: No restricted encryption or export-controlled components
- **Data Privacy**: No telemetry or outbound connections in core library
- **Security Scanning**: Compatible with enterprise security scanning tools
- **Compliance Frameworks**: Supports SOX, GDPR, HIPAA compliance requirements

## Dependency Management

### Reproducible Builds

All dependencies are locked to specific versions for deterministic builds:

```bash
# Install with locked versions for reproducibility
pip install -c constraints.txt -e .

# Verify installation
glassalpha --version
```

### Version Pinning Strategy

- **Conservative Ranges**: Dependencies use conservative version ranges in `pyproject.toml`
- **Exact Versions**: CI uses exact versions from `constraints.txt` for reproducible builds
- **Security Updates**: Regular updates for security patches while maintaining compatibility
- **Testing**: All version combinations tested in CI before release

## Support

### Open Source Support

- **Community Support**: GitHub Issues for bug reports and feature requests
- **Documentation**: Comprehensive documentation and examples
- **Contributing**: Community contributions welcome under Apache 2.0

## Compliance Considerations

### Regulatory Alignment

GlassAlpha's licensing approach aligns with major regulatory frameworks:

- **EU AI Act**: Transparency and auditability requirements met through open source licensing
- **CCPA/CPRA**: Consumer data protection supported by clean license structure
- **GDPR**: Data processing transparency enabled by source code availability
- **SOX Compliance**: Audit trail and change management supported

### Security & Privacy

- **No Telemetry**: Core library contains no outbound network calls or tracking
- **Local Processing**: All processing happens locally, no cloud dependencies
- **Data Protection**: No PII logging or data collection in audit processes
- **Enterprise Security**: Compatible with enterprise security policies and tools

---

For questions about enterprise deployments, or custom compliance solutions, please contact the GlassAlpha team.
