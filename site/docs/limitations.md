# Assumptions & Limitations

This document outlines the key assumptions and known limitations of Glass Alpha to ensure transparent and responsible use.

## Scope

### Supported Models
Glass Alpha is currently optimized for:
- **Tree-based models**: XGBoost, LightGBM, RandomForest, GradientBoosting
- **Linear models**: LogisticRegression, LinearRegression
- **Tabular data only**: Structured data with defined features

### Not Yet Supported
- Deep learning models (planned for v2.0)
- Computer vision models
- Natural language processing models
- Time series specific models (partial support via tabular features)
- Graph neural networks

## Technical Limitations

### Explainability

**TreeSHAP Limitations**
- Only exact for tree-based models
- Approximations used for other model types
- Computational complexity O(TLD²) where T=trees, L=leaves, D=depth

**Feature Interactions**
- Currently limited to 2-way interactions
- Higher-order interactions require exponential compute

**Background Data Requirements**
- SHAP requires representative background samples
- Poor background selection can bias explanations
- Recommended minimum: 100 samples for background

### Counterfactuals

**Feasibility Constraints**
- Cannot guarantee actionable counterfactuals for all instances
- Domain constraints must be manually specified
- Causal relationships must be provided, not inferred

**Search Limitations**
- Greedy search may miss global optima
- High-dimensional spaces (>100 features) may be slow
- Categorical features with many levels increase complexity

### Fairness Analysis

**Metric Limitations**
- No single metric captures all aspects of fairness
- Metrics can conflict with each other
- Requires predefined protected attributes

**Data Requirements**
- Needs sufficient samples per protected group (>30 recommended)
- Assumes protected attributes are available and accurate
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
- **Counterfactual search**: ~4GB RAM for 1000 searches
- **Audit report generation**: ~2GB RAM for standard report

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
