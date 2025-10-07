# Audit Report Contents

Complete breakdown of what's included in every GlassAlpha audit report.

---

## Visual walkthrough: anatomy of an audit report

This section shows you exactly what auditors, regulators, and validators look for in each section of a GlassAlpha audit report.

```mermaid
graph TD
    Report[GlassAlpha Audit Report] --> Header[📋 Report Header]
    Report --> Perf[📊 Performance Section]
    Report --> Fair[⚖️ Fairness Section]
    Report --> Explain[🔍 Explainability Section]
    Report --> Cal[🎯 Calibration Section]
    Report --> Manifest[🔒 Reproducibility Manifest]

    Header --> H1[Model ID & Version<br/>✓ Auditor checks: Unique identifier<br/>✓ Links to model registry]
    Header --> H2[Audit Timestamp<br/>✓ Auditor checks: When was this generated?<br/>✓ ISO 8601 format]
    Header --> H3[Compliance Profile<br/>✓ Auditor checks: Which regulations apply?<br/>✓ e.g., 'SR 11-7', 'GDPR Article 22']

    Perf --> P1[Confusion Matrix<br/>✓ Auditor checks: Error patterns by class<br/>✓ False negatives = missed good customers]
    Perf --> P2[Accuracy: 0.85<br/>✓ Auditor checks: Is this acceptable?<br/>✓ Banking: typically requires >80%]
    Perf --> P3[AUC-ROC: 0.92<br/>✓ Auditor checks: Discrimination ability<br/>✓ >0.80 is generally good]
    Perf --> P4[Precision/Recall<br/>✓ Auditor checks: Balance of errors<br/>✓ Which error type is worse?]

    Fair --> F1[Demographic Parity: 0.08<br/>✓ Auditor checks: <10% threshold<br/>✓ PASS: Within tolerance]
    Fair --> F2[Equal Opportunity: 0.12<br/>✓ Auditor checks: >10% threshold<br/>✓ FAIL: Qualified applicants treated unequally]
    Fair --> F3[Group Breakdown Table<br/>✓ Auditor checks: Sample sizes n≥30?<br/>✓ Which group is disadvantaged?]
    Fair --> F4[Statistical Significance<br/>✓ Auditor checks: p-values <0.05?<br/>✓ Is disparity real or noise?]

    Explain --> E1[Feature Importance Plot<br/>✓ Auditor checks: Top 10 features<br/>✓ Are protected attributes leaking?]
    Explain --> E2[SHAP Summary Plot<br/>✓ Auditor checks: Feature distributions<br/>✓ Positive vs negative contributions]
    Explain --> E3[Sample Explanations<br/>✓ Auditor checks: 3-5 representative cases<br/>✓ Do explanations make sense?]
    Explain --> E4[Method Documentation<br/>✓ Auditor checks: Which explainer used?<br/>✓ TreeSHAP, Coefficients, etc.]

    Cal --> C1[Calibration Curve<br/>✓ Auditor checks: Diagonal alignment<br/>✓ Predicted probabilities match reality?]
    Cal --> C2[Expected Calibration Error: 0.04<br/>✓ Auditor checks: <0.05 is good<br/>✓ >0.10 is concerning]
    Cal --> C3[Brier Score: 0.18<br/>✓ Auditor checks: Lower is better<br/>✓ Compare to baseline]
    Cal --> C4[Confidence Intervals<br/>✓ Auditor checks: Statistical uncertainty<br/>✓ Wide intervals = need more data]

    Manifest --> M1[Config Hash: sha256:abc123...<br/>✓ Auditor checks: Exact configuration used<br/>✓ Can reproduce with this hash]
    Manifest --> M2[Data Hash: sha256:def456...<br/>✓ Auditor checks: Which data version?<br/>✓ Tamper detection]
    Manifest --> M3[Random Seeds: [42, 123, 789]<br/>✓ Auditor checks: All randomness seeded?<br/>✓ Required for byte-identical reproduction]
    Manifest --> M4[Package Versions<br/>✓ Auditor checks: sklearn==1.3.0, etc.<br/>✓ Environment reproducibility]
    Manifest --> M5[Git Commit: 7f8a9b2<br/>✓ Auditor checks: Exact code version<br/>✓ Can inspect source code]

    style Header fill:#e1f5ff
    style Perf fill:#d4edda
    style Fair fill:#fff3cd
    style Explain fill:#e7d4f5
    style Cal fill:#ffd7e5
    style Manifest fill:#d1ecf1

    style F2 fill:#f8d7da
```

### What auditors look for: section-by-section checklist

#### 📋 Report Header (5 seconds)

**Critical checks**:

- ✅ Model ID matches registry entry
- ✅ Timestamp is recent (not stale analysis)
- ✅ Compliance profile matches regulatory requirement (e.g., SR 11-7 for credit)
- ✅ Author/submitter identified

**Red flags**:

- ❌ Missing model identifier
- ❌ Timestamp >6 months old
- ❌ Wrong compliance profile for use case
- ❌ No contact information

**Typical auditor question**: "Is this the same model deployed in production?"

---

#### 📊 Performance Section (30 seconds)

**Critical checks**:

- ✅ Accuracy >80% (banking/insurance standard)
- ✅ AUC-ROC >0.80 (discrimination ability)
- ✅ Confusion matrix shows acceptable error rates
- ✅ Test set size n≥1000 (statistical validity)

**Red flags**:

- ❌ Accuracy <70% (model not production-ready)
- ❌ AUC-ROC <0.70 (barely better than random)
- ❌ High false negative rate (missed good customers)
- ❌ Test set <100 samples (not statistically valid)

**Typical auditor questions**:

- "What's the cost of false positives vs false negatives?"
- "How does this compare to the baseline?"
- "Was this evaluated on held-out data?"

**What they're really checking**: Is the model accurate enough to justify its complexity and potential bias risks?

---

#### ⚖️ Fairness Section (2 minutes - most scrutinized)

**Critical checks**:

- ✅ Demographic parity <10% (ECOA standard for credit)
- ✅ Equal opportunity <10% (qualified applicants treated equally)
- ✅ Group sample sizes n≥30 (statistical validity)
- ✅ Statistical significance documented (p-values)
- ✅ Protected attributes identified (race, gender, age)

**Red flags**:

- ❌ Disparity >10% without business justification
- ❌ Sample size <30 for any group (unreliable)
- ❌ No intersectional analysis (single-attribute only)
- ❌ Missing statistical tests (could be sampling noise)
- ❌ Protected attributes used as model inputs (illegal for credit)

**Typical auditor questions**:

- "Why is there a 12% disparity in equal opportunity?"
- "What's the business justification for this difference?"
- "Have you tested intersectional bias (e.g., Black women)?"
- "What mitigation steps were attempted?"

**What they're really checking**: Does this model systematically disadvantage protected groups? If yes, can you justify it?

**Regulatory thresholds**:

- **ECOA/FCRA (Credit)**: <10% disparity in approval rates
- **EEOC (Employment)**: 80% rule (20% relative difference)
- **GDPR Article 22**: No automated decision if discriminatory
- **EU AI Act**: High-risk systems must demonstrate fairness testing

---

#### 🔍 Explainability Section (1 minute)

**Critical checks**:

- ✅ Top features make business sense (not spurious correlations)
- ✅ No protected attributes in top 10 (race, gender, etc.)
- ✅ Explainer method documented (TreeSHAP, Coefficients, etc.)
- ✅ Sample explanations provided (3-5 representative cases)
- ✅ Feature contributions sum to prediction (additivity property)

**Red flags**:

- ❌ Top feature is zip code (proxy for race)
- ❌ Protected attributes appear in feature importance
- ❌ Explainer method not documented
- ❌ Explanations don't match business logic
- ❌ SHAP values don't satisfy additivity

**Typical auditor questions**:

- "Why is 'first name' the #2 most important feature?" (gender proxy)
- "Can you explain this specific denial to the applicant?"
- "How do you know these explanations are accurate?"

**What they're really checking**: Are the model's reasons legitimate business factors, or is it learning to discriminate via proxies?

**Regulatory requirements**:

- **SR 11-7**: Model must be explainable to validators
- **ECOA**: Adverse action notices must cite specific reasons
- **GDPR Article 22**: Right to explanation for automated decisions
- **EU AI Act**: High-risk systems must provide explanations

---

#### 🎯 Calibration Section (30 seconds)

**Critical checks**:

- ✅ Expected Calibration Error (ECE) <0.05 (well-calibrated)
- ✅ Calibration curve close to diagonal (predicted = actual)
- ✅ Brier score reported (lower is better)
- ✅ Confidence intervals provided (statistical uncertainty)

**Red flags**:

- ❌ ECE >0.10 (poorly calibrated)
- ❌ Calibration curve far from diagonal (over/under-confident)
- ❌ No confidence intervals (can't assess reliability)
- ❌ Calibration not tested per group (could be calibrated overall but not within groups)

**Typical auditor questions**:

- "If the model says 80% probability, is it right 80% of the time?"
- "Is calibration consistent across demographic groups?"

**What they're really checking**: Can we trust the model's confidence scores for high-stakes decisions?

**Why calibration matters**:

- **Insurance pricing**: Premiums based on predicted probabilities
- **Credit scoring**: Interest rates tied to default probability
- **Healthcare**: Treatment decisions based on risk scores

---

#### 🔒 Reproducibility Manifest (15 seconds)

**Critical checks**:

- ✅ Config hash provided (can reproduce exact run)
- ✅ Data hash provided (tamper detection)
- ✅ Random seeds documented (all sources of randomness)
- ✅ Package versions listed (environment reproducibility)
- ✅ Git commit SHA provided (exact code version)

**Red flags**:

- ❌ No config hash (can't verify exact settings)
- ❌ No random seeds (non-reproducible)
- ❌ Missing package versions (environment drift risk)
- ❌ No git commit (can't inspect source code)
- ❌ Timestamp mismatch (manifest generated at different time)

**Typical auditor questions**:

- "Can I reproduce this audit byte-for-byte?"
- "What happens if I run this again?"
- "How do I know this data wasn't tampered with?"

**What they're really checking**: Is this audit trustworthy? Can it be independently validated?

**Regulatory requirements**:

- **SR 11-7**: Model validation must be reproducible
- **FDA (Medical Devices)**: Clinical validation must be reproducible
- **EU AI Act**: High-risk systems must maintain audit trails

---

### Common auditor questions by role

#### Compliance officer (regulatory risk)

1. "Does this pass our fairness thresholds?" → Check Fairness Section
2. "Can we defend this to CFPB/EEOC/FDA?" → Check Fairness + Explainability
3. "Is there documentation for legal?" → Check Manifest + Config Hash
4. "What's our exposure if we deploy this?" → Check Red Flags across all sections

#### Model validator (technical verification)

1. "Is the accuracy acceptable?" → Check Performance Section
2. "Are the explanations correct?" → Check Explainability + SHAP additivity
3. "Can I reproduce this?" → Check Manifest + Seeds + Hashes
4. "What are the model limitations?" → Check Sample Sizes + Confidence Intervals

#### Risk manager (business impact)

1. "What error rate can we tolerate?" → Check Confusion Matrix + Cost Analysis
2. "Which demographic groups are affected?" → Check Fairness Group Breakdown
3. "What's the worst-case scenario?" → Check Maximum Disparity + Regulatory Thresholds
4. "Do we need to retrain or mitigate?" → Check FAIL flags + Justifications

#### External auditor/regulator (independent verification)

1. "Show me the evidence pack." → Manifest + Config + Data Hashes
2. "Can you reproduce this in front of me?" → Seeds + Git Commit + Package Versions
3. "Explain this 12% fairness violation." → Fairness Section + Business Justification
4. "Why should I trust these explanations?" → Explainer Documentation + Validation

---

### How to use this guide

**Before submitting an audit**:

1. Go through each section with the auditor checklist
2. Mark ✅ for items that pass, ❌ for items that fail
3. Prepare justifications for any ❌ red flags
4. Ensure Manifest section is complete (reproducibility is critical)

**When reviewing someone else's audit**:

1. Start with Fairness Section (highest regulatory risk)
2. Check Manifest (can you reproduce it?)
3. Validate Performance (is accuracy acceptable?)
4. Review Explainability (do reasons make sense?)
5. Verify Calibration (can you trust probabilities?)

**For regulatory submission**:

1. Generate evidence pack (PDF + Manifest + Config + Hashes)
2. Prepare 1-page summary for compliance officer
3. Document all red flags with justifications
4. Include independent validator sign-off
5. Archive for retention period (typically 7 years)

---

## 1. Model performance metrics

Every audit includes comprehensive performance evaluation:

- **Classification metrics**: Accuracy, precision, recall, F1 score, AUC-ROC
- **Confusion matrices**: Visual breakdown of true/false positives and negatives
- **Performance curves**: ROC curves and precision-recall curves
- **Cross-validation results**: Statistical validation of model stability

These metrics provide the foundation for understanding model behavior and are required by most regulatory frameworks.

[See configuration guide →](../getting-started/configuration.md)

## 2. Model explanations

Understanding why models make specific predictions:

### Feature importance

- **For linear models**: Coefficient-based explanations (zero dependencies)
- **For tree models**: SHAP (SHapley Additive exPlanations) values
- **Visual rankings**: Clear ordering of most impactful features

### Individual predictions

- **Per-prediction breakdown**: Feature contributions to specific decisions
- **Visual explanations**: Force plots showing positive/negative influences
- **Deterministic ranking**: Consistent ordering across runs

[See explainer selection guide →](../reference/explainers.md)

## 3. Fairness analysis

Comprehensive bias detection across demographic groups:

### Group fairness

- **Demographic parity**: Equal positive prediction rates across groups
- **Equal opportunity**: Equal true positive rates across groups
- **Statistical confidence**: Confidence intervals for all fairness metrics

### Intersectional fairness

- **Multi-attribute analysis**: Combined effects of multiple protected attributes
- **Subgroup detection**: Identification of particularly affected intersections

### Individual fairness

- **Consistency testing**: Similar individuals receive similar predictions
- **Matched pairs analysis**: Direct comparison of similar cases
- **Disparate treatment detection**: Identification of inconsistent decisions

[See fairness metrics reference →](fairness-metrics.md)

## 4. Calibration analysis

Model confidence accuracy evaluation:

- **Calibration curves**: Visual representation of prediction reliability
- **Expected Calibration Error (ECE)**: Quantitative calibration quality
- **Brier score**: Comprehensive probability accuracy measure
- **Confidence intervals**: Statistical bounds on calibration metrics

Calibration is critical for high-stakes decisions where probability estimates matter.

[See calibration reference →](calibration.md)

## 5. Robustness testing

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

## 7. Dataset bias detection

Pre-model bias identification:

- **Proxy correlation analysis**: Identification of protected attribute proxies
- **Distribution drift**: Changes in demographic composition
- **Class imbalance**: Detection of underrepresented groups

[See dataset bias guide →](../guides/dataset-bias.md)

## 8. Preprocessing verification

Production artifact validation:

- **File hash**: SHA256 fingerprint of preprocessing pipeline
- **Params hash**: Canonical hash of learned parameters
- **Version compatibility**: Runtime environment verification
- **Class allowlisting**: Security validation against pickle exploits

[See preprocessing guide →](../guides/preprocessing.md)

## 9. Reproducibility manifest

Complete audit trail for regulatory submission:

### Configuration hash

- **Complete config fingerprint**: SHA256 of entire configuration
- **Policy version**: Specific compliance rules applied
- **Profile used**: Audit profile and feature set

### Dataset fingerprint

- **Data hash**: Cryptographic hash of input data
- **Schema lock**: Structure and column validation
- **Sample size**: Number of records processed

### Runtime environment

- **Git commit SHA**: Exact code version used
- **Timestamp**: ISO 8601 formatted execution time
- **Package versions**: All dependencies with versions
- **Random seeds**: All seeds used for reproducibility

### Model artifacts

- **Model hash**: Fingerprint of trained model
- **Preprocessing hash**: Hash of preprocessing artifacts
- **Feature list**: Exact features used

This manifest enables byte-identical reproduction of the audit on the same inputs.

[See determinism guide →](../guides/preprocessing.md#determinism)

## Example audit

See a complete audit in action:

- [German Credit Audit](../examples/german-credit-audit.md) - Full walkthrough with credit scoring
- [Healthcare Bias Detection](../examples/healthcare-bias-detection.md) - Medical AI compliance
- [Fraud Detection Audit](../examples/fraud-detection-audit.md) - Financial services compliance

## Regulatory mapping

See how these components map to specific regulatory requirements:

- [SR 11-7 Technical Mapping](../compliance/sr-11-7-mapping.md) - Federal Reserve guidance for banking
- [Trust & Deployment](trust-deployment.md) - Architecture and compliance overview
