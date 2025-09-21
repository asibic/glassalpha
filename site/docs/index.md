# Glass Alpha

## Open-core AI compliance toolkit for tabular ML

Glass Alpha provides enterprise-grade explainability, fairness, and compliance tools for tabular machine learning models. Built with a focus on regulatory requirements and on-premise deployment.

!!! success "60-Second Quick Start"
    ```python
    from glassalpha import explain, audit
    import xgboost as xgb
    
    # Train your model
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)
    
    # Generate explanations
    explanations = explain(model, X_test)
    
    # Create audit report
    report = audit(model, X_test, y_test, output="audit_report.pdf")
    ```
    **That's it!** You now have a regulator-ready PDF audit report.

## Key Features

<div class="grid cards" markdown>

-   :material-eye-check:{ .lg .middle } **Explainability**

    ---

    TreeSHAP-based feature importance and individual predictions explanations for tree-based models

    [:octicons-arrow-right-24: Learn more](features/explainability.md)

-   :material-scale-balance:{ .lg .middle } **Fairness Analysis**

    ---

    Comprehensive bias detection and mitigation across protected attributes

    [:octicons-arrow-right-24: Learn more](features/fairness.md)

-   :material-file-document-check:{ .lg .middle } **Audit Reports**

    ---

    Deterministic, regulator-ready PDF reports with full reproducibility

    [:octicons-arrow-right-24: Learn more](features/audit.md)

-   :material-sync:{ .lg .middle } **Counterfactual Recourse**

    ---

    Causal-aware, feasible recommendations for changing model decisions

    [:octicons-arrow-right-24: Learn more](features/counterfactuals.md)

</div>

## Why Glass Alpha?

### 🏢 Enterprise-Ready
- **On-premise first**: No external API calls, runs entirely offline
- **Deterministic outputs**: Seeded runs produce identical results
- **Policy-as-code**: Codify compliance requirements in YAML

### 📊 Built for Tabular ML
- Optimized for XGBoost, LightGBM, scikit-learn
- Handles real-world messy data
- Scales to millions of rows

### ✅ Regulatory Compliance
- GDPR Article 22 compliant explanations
- Fair lending (ECOA/FCRA) compatible
- Audit trail with immutable manifests

## Supported Models

| Framework | Status | Notes |
|-----------|--------|-------|
| XGBoost | ✅ Full support | TreeSHAP optimized |
| LightGBM | ✅ Full support | Native integration |
| scikit-learn | ✅ Full support | Random Forest, Logistic Regression |
| CatBoost | 🔄 Coming soon | Q2 2025 |
| Deep Learning | 📅 Planned | v2.0 roadmap |

## Installation

=== "Standard"
    ```bash
    pip install glassalpha
    ```

=== "Development"
    ```bash
    git clone https://github.com/GlassAlpha/glassalpha
    cd glassalpha
    pip install -e packages/[dev]
    ```

=== "Enterprise"
    ```bash
    # For on-premise deployment with additional security features
    pip install glassalpha[enterprise]
    ```

## Quick Examples

### Generate Explanations
```python
from glassalpha import Explainer
import pandas as pd

explainer = Explainer(model)
shap_values = explainer.explain(X_test)

# Feature importance
explainer.plot_importance()

# Individual prediction explanation
explainer.plot_waterfall(X_test.iloc[0])
```

### Detect Bias
```python
from glassalpha import FairnessAnalyzer

analyzer = FairnessAnalyzer(
    protected_attributes=['gender', 'race']
)

bias_report = analyzer.analyze(
    model, X_test, y_test, y_pred
)

print(bias_report.disparate_impact)
```

### Generate Audit Report
```python
from glassalpha import AuditReport

report = AuditReport(
    model=model,
    data=(X_test, y_test),
    config="configs/audit_config.yaml"
)

report.generate("audit_report_2025.pdf")
```

## Next Steps

- 📚 [Read the Getting Started Guide](getting-started/quickstart.md)
- 💡 [Explore Example Notebooks](examples/german-credit.md)
- 🛠️ [API Reference](api/overview.md)
- 👥 [Contributing Guidelines](contributing.md)

## License

Glass Alpha is released under the Apache 2.0 License. See [LICENSE](https://github.com/GlassAlpha/glassalpha/blob/main/LICENSE) for details.

## Support

- 📧 Email: support@glassalpha.io
- 💬 GitHub Discussions: [GlassAlpha/glassalpha/discussions](https://github.com/GlassAlpha/glassalpha/discussions)
- 🐛 Issues: [GlassAlpha/glassalpha/issues](https://github.com/GlassAlpha/glassalpha/issues)
