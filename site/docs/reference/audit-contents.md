# Audit Report Contents

Complete breakdown of what's included in every GlassAlpha audit report.

## 1. Model Performance Metrics

Every audit includes comprehensive performance evaluation:

- **Classification metrics**: Accuracy, precision, recall, F1 score, AUC-ROC
- **Confusion matrices**: Visual breakdown of true/false positives and negatives
- **Performance curves**: ROC curves and precision-recall curves
- **Cross-validation results**: Statistical validation of model stability

These metrics provide the foundation for understanding model behavior and are required by most regulatory frameworks.

[See configuration guide →](../getting-started/configuration.md)

## 2. Model Explanations

Understanding why models make specific predictions:

### Feature Importance

- **For linear models**: Coefficient-based explanations (zero dependencies)
- **For tree models**: SHAP (SHapley Additive exPlanations) values
- **Visual rankings**: Clear ordering of most impactful features

### Individual Predictions

- **Per-prediction breakdown**: Feature contributions to specific decisions
- **Visual explanations**: Force plots showing positive/negative influences
- **Deterministic ranking**: Consistent ordering across runs

[See explainer selection guide →](../reference/explainers.md)

## 3. Fairness Analysis

Comprehensive bias detection across demographic groups:

### Group Fairness

- **Demographic parity**: Equal positive prediction rates across groups
- **Equal opportunity**: Equal true positive rates across groups
- **Statistical confidence**: Confidence intervals for all fairness metrics

### Intersectional Fairness

- **Multi-attribute analysis**: Combined effects of multiple protected attributes
- **Subgroup detection**: Identification of particularly affected intersections

### Individual Fairness

- **Consistency testing**: Similar individuals receive similar predictions
- **Matched pairs analysis**: Direct comparison of similar cases
- **Disparate treatment detection**: Identification of inconsistent decisions

[See fairness metrics reference →](fairness-metrics.md)

## 4. Calibration Analysis

Model confidence accuracy evaluation:

- **Calibration curves**: Visual representation of prediction reliability
- **Expected Calibration Error (ECE)**: Quantitative calibration quality
- **Brier score**: Comprehensive probability accuracy measure
- **Confidence intervals**: Statistical bounds on calibration metrics

Calibration is critical for high-stakes decisions where probability estimates matter.

[See calibration reference →](calibration.md)

## 5. Robustness Testing

Adversarial perturbation analysis:

- **Epsilon sweeps**: Model behavior under small input changes
- **Feature perturbations**: Individual feature stability testing
- **Robustness score**: Quantitative measure of model stability

[See robustness reference →](robustness.md)

## 6. Reason codes (ECOA compliance)

For credit decisions, adverse action notice generation:

- **Top-N negative contributions**: Features that hurt the applicant's score
- **ECOA-compliant formatting**: Regulatory-ready adverse action notices
- **Protected attribute exclusion**: Automatic removal of prohibited factors
- **Deterministic ranking**: Consistent reason codes across runs

[See reason codes guide →](../guides/reason-codes.md)

## 7. Dataset Bias Detection

Pre-model bias identification:

- **Proxy correlation analysis**: Identification of protected attribute proxies
- **Distribution drift**: Changes in demographic composition
- **Class imbalance**: Detection of underrepresented groups

[See dataset bias guide →](../guides/dataset-bias.md)

## 8. Preprocessing Verification

Production artifact validation:

- **File hash**: SHA256 fingerprint of preprocessing pipeline
- **Params hash**: Canonical hash of learned parameters
- **Version compatibility**: Runtime environment verification
- **Class allowlisting**: Security validation against pickle exploits

[See preprocessing guide →](../guides/preprocessing.md)

## 9. Reproducibility Manifest

Complete audit trail for regulatory submission:

### Configuration Hash

- **Complete config fingerprint**: SHA256 of entire configuration
- **Policy version**: Specific compliance rules applied
- **Profile used**: Audit profile and feature set

### Dataset Fingerprint

- **Data hash**: Cryptographic hash of input data
- **Schema lock**: Structure and column validation
- **Sample size**: Number of records processed

### Runtime Environment

- **Git commit SHA**: Exact code version used
- **Timestamp**: ISO 8601 formatted execution time
- **Package versions**: All dependencies with versions
- **Random seeds**: All seeds used for reproducibility

### Model Artifacts

- **Model hash**: Fingerprint of trained model
- **Preprocessing hash**: Hash of preprocessing artifacts
- **Feature list**: Exact features used

This manifest enables byte-identical reproduction of the audit on the same inputs.

[See determinism guide →](../guides/preprocessing.md#determinism)

## Example Audit

See a complete audit in action:

- [German Credit Audit](../examples/german-credit-audit.md) - Full walkthrough with credit scoring
- [Healthcare Bias Detection](../examples/healthcare-bias-detection.md) - Medical AI compliance
- [Fraud Detection Audit](../examples/fraud-detection-audit.md) - Financial services compliance

## Regulatory Mapping

See how these components map to specific regulatory requirements:

- [SR 11-7 Technical Mapping](../compliance/sr-11-7-mapping.md) - Federal Reserve guidance for banking
- [Trust & Deployment](trust-deployment.md) - Architecture and compliance overview
