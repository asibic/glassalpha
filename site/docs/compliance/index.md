# Compliance Overview

Welcome to GlassAlpha compliance documentation. This guide helps you find the right resources for your role and industry.

## Quick Navigation

### By Industry

Choose your industry for specific regulatory guidance:

- **[Banking & Credit](banking-guide.md)** - SR 11-7, ECOA, FCRA compliance
- **[Insurance](insurance-guide.md)** - NAIC Model Act #670, rate fairness
- **[Healthcare](healthcare-guide.md)** - HIPAA, health equity mandates
- **[Fraud Detection](fraud-guide.md)** - FCRA adverse action, FTC fairness

### By Role

Choose your role for workflow-specific guidance:

- **[ML Engineers](../guides/ml-engineer-workflow.md)** - Implementation, CI integration, debugging
- **[Compliance Officers](../guides/compliance-workflow.md)** - Evidence packs, policy gates, regulator communication
- **[Model Validators](../guides/validator-workflow.md)** - Verification, challenge, independent review

## Decision Tree

Not sure where to start? Follow this decision tree:

```
┌─────────────────────────────────────┐
│ What do you need to accomplish?    │
└─────────────────────────────────────┘
                 │
        ┌────────┴─────────┐
        │                  │
    Implement          Verify/Review
    an audit           an audit
        │                  │
        ▼                  ▼
┌───────────────┐    ┌───────────────┐
│ ML Engineer   │    │ Compliance    │
│ Workflow      │    │ Officer or    │
│               │    │ Validator     │
└───────────────┘    └───────────────┘
        │                  │
        │            ┌─────┴──────┐
        │            │            │
        │         Submit to   Independent
        │         regulator   review
        │            │            │
        │            ▼            ▼
        │      ┌──────────┐  ┌──────────┐
        │      │Compliance│  │Validator │
        │      │Workflow  │  │Workflow  │
        │      └──────────┘  └──────────┘
        │
        ▼
┌─────────────────────────────────────┐
│ What industry?                      │
├─────────────────────────────────────┤
│ • Banking/Credit → Banking Guide    │
│ • Insurance → Insurance Guide       │
│ • Healthcare → Healthcare Guide     │
│ • Fraud Detection → Fraud Guide     │
│ • Other → Quickstart Guide          │
└─────────────────────────────────────┘
```

## Common Scenarios

### Scenario 1: "I need to pass an SR 11-7 audit"

**Your role**: Compliance officer or risk manager at a bank

**Path**:

1. Start with [Banking Compliance Guide](banking-guide.md)
2. Review [SR 11-7 Technical Mapping](sr-11-7-mapping.md) for clause-by-clause coverage
3. Work with ML team using [ML Engineer Workflow](../guides/ml-engineer-workflow.md)
4. Generate evidence pack using [Compliance Officer Workflow](../guides/compliance-workflow.md)

**Key artifacts**: Audit PDF, evidence pack, policy decision log

### Scenario 2: "I need to integrate audits into CI/CD"

**Your role**: ML engineer or data scientist

**Path**:

1. Start with [ML Engineer Workflow](../guides/ml-engineer-workflow.md)
2. Review industry-specific requirements ([Banking](banking-guide.md) / [Insurance](insurance-guide.md) / [Healthcare](healthcare-guide.md))
3. Set up policy gates with [Compliance Workflow](../guides/compliance-workflow.md)
4. Implement pre-commit hooks or GitHub Actions

**Key features**: Policy gates, CLI automation, deterministic outputs

### Scenario 3: "I need to validate someone else's audit"

**Your role**: Internal auditor, model validator, third-party consultant

**Path**:

1. Start with [Model Validator Workflow](../guides/validator-workflow.md)
2. Review relevant industry guide for regulatory context
3. Verify evidence pack integrity
4. Challenge findings using checklist

**Key features**: Evidence pack verification, reproducibility checks, red flag detection

### Scenario 4: "I need to explain a credit denial"

**Your role**: Compliance officer responding to consumer inquiry

**Path**:

1. Review [Banking Compliance Guide](banking-guide.md) - ECOA requirements
2. Generate reason codes with [Reason Codes Guide](../guides/reason-codes.md)
3. Optionally provide recourse with [Recourse Guide](../guides/recourse.md)

**Key artifacts**: Adverse action notice with specific reasons

### Scenario 5: "I need to test model robustness"

**Your role**: Risk manager or model validator

**Path**:

1. Review [Shift Testing Guide](../guides/shift-testing.md)
2. Apply demographic shift scenarios
3. Document results in audit report

**Key features**: `--check-shift` flag, stress testing, scenario analysis

## Regulatory Framework Coverage

### Banking & Finance

- **SR 11-7** (Federal Reserve): Model risk management
- **ECOA** (CFPB): Equal credit opportunity
- **FCRA** (FTC): Fair credit reporting
- **Fair Lending Laws**: Anti-discrimination requirements

**See**: [Banking Compliance Guide](banking-guide.md)

### Insurance

- **NAIC Model Act #670**: Prohibition on unfair discrimination
- **State regulations**: Varies by jurisdiction (CA, NY, etc.)
- **Anti-discrimination laws**: Protected characteristics in underwriting

**See**: [Insurance Compliance Guide](insurance-guide.md)

### Healthcare

- **HIPAA**: Privacy and security of health information
- **Health equity mandates**: CMS quality measures, state requirements
- **Clinical validation**: IRB requirements, informed consent

**See**: [Healthcare Compliance Guide](healthcare-guide.md)

### Cross-Industry

- **GDPR Article 22**: Right to explanation (EU)
- **AI Act** (EU): High-risk AI systems
- **FTC guidance**: Algorithmic fairness, consumer protection
- **CCPA** (California): Consumer privacy rights

**See**: Industry-specific guides for details

## Core Capabilities

### Audit Reports

Comprehensive PDF reports covering:

- Model documentation and validation testing
- Performance metrics with statistical confidence intervals
- Fairness analysis (group and individual)
- Calibration testing (predicted vs actual outcomes)
- Explainability (SHAP values, feature contributions)
- Reason codes (ECOA-compliant adverse action notices)
- Recourse analysis (counterfactual recommendations)
- Dataset bias detection
- Robustness testing (demographic shifts, adversarial perturbations)

### Evidence Packs

Tamper-evident zip files containing:

- Audit PDF
- Provenance manifest (hashes, versions, seeds)
- Policy decision log (pass/fail for each gate)
- Configuration files
- Dataset schema
- SHA256 checksums for integrity verification

### Policy-as-Code Gates

Define compliance thresholds in YAML:

- Minimum calibration accuracy
- Maximum fairness metric values
- Required sample sizes
- Robustness requirements

Automatically fail non-compliant models in CI/CD.

### Reproducibility

Byte-identical outputs under same conditions:

- Explicit random seeds
- Package version tracking
- Data hashing (SHA256)
- Git commit tracking
- Platform-independent determinism

## Getting Started

### For First-Time Users

1. **Install**: `pip install glassalpha`
2. **Quickstart**: [60-second audit tutorial](../getting-started/quickstart.md)
3. **Choose path**: Industry guide or role workflow
4. **Run audit**: `glassalpha audit --config audit.yaml --output report.pdf`

### For Experienced Users

- [CLI Reference](../reference/cli.md) - All commands and options
- [Configuration Guide](../getting-started/configuration.md) - Advanced config
- [Troubleshooting](../reference/troubleshooting.md) - Common issues

## Support

### Documentation

- **Getting Started**: [Installation](../getting-started/installation.md) | [Quickstart](../getting-started/quickstart.md) | [Configuration](../getting-started/configuration.md)
- **Examples**: [German Credit](../examples/german-credit-audit.md) | [Healthcare Bias](../examples/healthcare-bias-detection.md) | [Fraud Detection](../examples/fraud-detection-audit.md)
- **Reference**: [CLI](../reference/cli.md) | [Fairness Metrics](../reference/fairness-metrics.md) | [Calibration](../reference/calibration.md)

### Community

- **GitHub**: [GlassAlpha/glassalpha](https://github.com/GlassAlpha/glassalpha)
- **Discussions**: [Ask questions, share use cases](https://github.com/GlassAlpha/glassalpha/discussions)
- **Issues**: [Report bugs, request features](https://github.com/GlassAlpha/glassalpha/issues)

### Contact

- **Email**: [contact@glassalpha.com](mailto:contact@glassalpha.com)
- **Website**: [glassalpha.com](https://glassalpha.com)

## Next Steps

Choose your path:

- **Banking teams** → [Banking Compliance Guide](banking-guide.md)
- **Insurance teams** → [Insurance Compliance Guide](insurance-guide.md)
- **Healthcare teams** → [Healthcare Compliance Guide](healthcare-guide.md)
- **ML engineers** → [ML Engineer Workflow](../guides/ml-engineer-workflow.md)
- **Compliance officers** → [Compliance Officer Workflow](../guides/compliance-workflow.md)
- **Model validators** → [Model Validator Workflow](../guides/validator-workflow.md)
