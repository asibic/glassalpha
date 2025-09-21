# Phase 1 Scope & Limitations

This document outlines the Phase 1 scope, assumptions, and limitations of Glass Alpha's **audit-first** approach to ensure transparent and responsible use.

## Phase 1 Audit-First Scope

!!! info "Phase 1 Focus: PDF Audit Reports Only"
    Phase 1 delivers **one core capability**: deterministic, regulator-ready PDF audit reports. All other features are minimal POCs or deferred to Phase 2.

### Supported Models (Phase 1)
- **XGBoost** ✅ Full audit support
- **LightGBM** ✅ Full audit support  
- **Logistic Regression** ✅ Full audit support
- **Random Forest** 🔄 Coming Q4 2025

### Phase 1 Audit Components
- **Model Performance**: Accuracy, precision, recall, F1, AUC-ROC, confusion matrices
- **TreeSHAP Explanations**: Feature importance, individual predictions, waterfall plots
- **Basic Fairness Analysis**: Minimal POC with protected attribute analysis
- **Reproducibility Manifest**: Complete lineage tracking (config hash, data hash, git SHA, seeds)

### Not in Phase 1
- Deep learning models → Phase 2
- Advanced fairness monitoring → Phase 2
- Counterfactual explanations → Phase 2
- Drift detection (beyond basic PSI) → Phase 2
- Dashboard or web interfaces → Phase 2
- API services → Phase 2

## Phase 1 Technical Limitations

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


### Basic Fairness Analysis (Phase 1 POC)

**Limited Scope**
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
- Violations may affect counterfactual validity
- Correlation matrices should be reviewed

### Distribution Assumptions
- Some fairness metrics assume specific distributions
- Drift detection assumes stationarity within windows
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
- Internationalization planned for future releases

## Known Issues

### Current Bugs
- Memory leak in repeated counterfactual generation (fix in 0.2.1)
- Slow performance on sparse matrices (optimization planned)
- PDF generation fails with non-ASCII characters (workaround available)

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

## Future Improvements

### Roadmap
- Deep learning support (v2.0)
- Real-time explanation serving (v1.5)
- Causal discovery automation (v2.5)
- Multi-language support (v2.0)

### Feedback
We welcome feedback on limitations and feature requests:
- GitHub Issues: [Report limitations](https://github.com/GlassAlpha/glassalpha/issues)
- Discussions: [Request features](https://github.com/GlassAlpha/glassalpha/discussions)

## Changelog

### Version 0.1.0 (Current)
- Initial release with core functionality
- Known limitations documented above

---

*Last updated: September 2025*
