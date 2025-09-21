# Scope & Limitations

This document outlines the current scope, assumptions, and limitations of Glass Alpha's **audit-first** approach to ensure transparent and responsible use.

## Audit-First Scope

!!! info "Current Focus: PDF Audit Reports"
    Glass Alpha delivers **one core capability**: deterministic, regulator-ready PDF audit reports. Additional features planned for future releases.

### Currently Supported Models
- **XGBoost** ✅ Full audit support
- **LightGBM** ✅ Full audit support  
- **Logistic Regression** ✅ Full audit support

### Current Audit Components
- **Model Performance**: Accuracy, precision, recall, F1, AUC-ROC, confusion matrices
- **TreeSHAP Explanations**: Feature importance, individual predictions, waterfall plots
- **Basic Fairness Analysis**: Protected attribute analysis and group parity metrics
- **Reproducibility Manifest**: Complete lineage tracking (config hash, data hash, git SHA, seeds)

### Not Currently Supported
- Deep learning models
- Advanced fairness monitoring beyond basic metrics
- Counterfactual explanations
- Continuous drift detection
- Dashboard or web interfaces
- API services

## Technical Limitations

### PDF Audit Generation

**Deterministic Output Requirements**
- Requires identical Python/package versions for byte-identical PDFs
- Git repository required for commit SHA tracking
- Seeded random number generation throughout pipeline

**TreeSHAP Explanations**
- Only exact for tree-based models (XGBoost, LightGBM)
- Approximations used for Logistic Regression
- Computational complexity O(TLD²) where T=trees, L=leaves, D=depth
- Background data selection affects explanation quality


### Basic Fairness Analysis

**Current Limitations**
- Basic group parity metrics only (disparate impact, equal opportunity difference)
- No advanced bias mitigation techniques
- Minimal statistical testing
- Simple protected attribute analysis

**Data Requirements**
- Requires predefined protected attributes
- Needs sufficient samples per group (>30 recommended)
- Cannot detect unlabeled proxy discrimination

## Performance Considerations

### Scalability

| Operation | Complexity | Practical Limit |
|-----------|-----------|-----------------|
| SHAP values | O(n_samples × TLD²) | 1M samples |
| Counterfactuals | O(n_features²) | 200 features |
| Fairness metrics | O(n_samples × n_groups) | 100K samples |
| Audit report | O(n_samples × n_features) | 10M cells |

### Memory Requirements

- **SHAP computation**: ~8GB RAM per 100K samples with 100 features
- **PDF report generation**: ~1GB RAM for standard audit
- **Reproducibility tracking**: ~100MB for manifest data

## Statistical Assumptions

### Independence Assumptions
- Features are assumed independent for some calculations
- Correlation matrices should be reviewed

### Distribution Assumptions
- Some fairness metrics assume specific distributions
- Confidence intervals assume sufficient sample sizes

## Regulatory Considerations

### Compliance Limitations

**Not Legal Advice**
- Glass Alpha provides technical tools, not legal guidance
- Compliance depends on jurisdiction and use case
- Consult legal experts for regulatory requirements

**Audit Reports**
- Reports are informative, not certifications
- Human review required for critical decisions
- Local regulations may require additional documentation

### Geographic Limitations

**Data Residency**
- No built-in data residency controls
- Users responsible for data governance
- On-premise deployment recommended for sensitive data

**Language Support**
- Documentation and reports in English only

## Known Issues

### Current Issues
- Slow performance on sparse matrices
- PDF generation may fail with non-ASCII characters in feature names

### Edge Cases
- Models with >1000 trees may timeout
- Extreme class imbalance (<1% minority) affects metrics
- Missing value handling inconsistent across modules

## Responsible AI Considerations

### Ethical Limitations

**Bias Amplification**
- Cannot eliminate all forms of bias
- May reflect historical discrimination in data
- Requires ongoing monitoring and adjustment

**Explanation Misuse**
- Explanations can be gamed if exposed to adversaries
- Should not be sole basis for high-stakes decisions
- Consider explanation uncertainty and limitations

### Human-in-the-Loop

**Required Human Oversight**
- Critical decisions should involve human review
- Explanations are aids, not replacements for judgment
- Domain expertise necessary for interpretation

## Recommendations

### Best Practices

1. **Validate assumptions** before deployment
2. **Monitor performance** continuously in production
3. **Document limitations** for end users
4. **Implement fallbacks** for edge cases
5. **Regular audits** of model decisions

### When Not to Use Glass Alpha

❌ **Inappropriate use cases:**
- Life-critical systems without human oversight
- Fully automated decision-making for protected classes
- Systems requiring real-time (<10ms) explanations
- Non-tabular data without feature engineering

✅ **Appropriate use cases:**
- Decision support systems with human review
- Model development and debugging
- Regulatory compliance documentation
- Bias detection and monitoring

## Community & Feedback

We welcome contributions and feedback:
- GitHub Issues: [Report issues](https://github.com/GlassAlpha/glassalpha/issues)
- Discussions: [Community discussions](https://github.com/GlassAlpha/glassalpha/discussions)

Future improvements may come through ongoing development.

---

*Documentation last updated: September 2025*
