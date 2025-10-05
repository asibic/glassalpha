# GlassAlpha

An ([open source](https://glassalpha.com/reference/trust-deployment/#licensing-dependencies)) toolkit to generate deterministic, regulator-ready PDF audit reports for tabular ML models.

_Note: GlassAlpha is currently pre-alpha while I’m still making significant changes. I’ll cut the first official release and publish it on PyPI once things stabilize. The audits do run and the package works, so feel free to try it out, feedback welcome!_

## Get started

### Run your first audit in 30 seconds

Clone and install

```bash
git clone https://github.com/GlassAlpha/glassalpha
cd glassalpha/packages
pip install -e ".[all]"  # Install with all optional features
```

Create a configuration (interactive wizard)

```bash
glassalpha init
```

Generate an audit report

```bash
glassalpha audit  # Uses smart defaults - no flags needed!
```

That's it. You now have a complete audit report with model performance, SHAP explanations, and fairness metrics.

**Tip:** Run `glassalpha doctor` anytime to check what features are available and see installation options.

**More details:** See the [full installation guide](packages/README.md#installation) and [German Credit tutorial](https://glassalpha.com/examples/german-credit-audit/) to see what's in the report.

## Structure

- **`packages/`** - The actual Python package ([dev docs here](packages/README.md))
- **`site/`** - User documentation and tutorials. The docs site is at [glassalpha.com](https://glassalpha.com/)
- **`configs/`** - Example audit configs you can copy and modify

## Capabilities

Right now, GlassAlpha handles:

- **Models**: XGBoost, LightGBM, Logistic Regression (more coming)
- **Explanations**: TreeSHAP feature importance with individual prediction breakdowns
- **Fairness**: Demographic parity, equal opportunity, bias detection
- **Output**: Professional PDFs and HTML reports that are byte-identical on repeat runs
- **Everything runs locally** - your data never leaves your machine
- **CI/CD Ready**: JSON error output and standardized exit codes for automation

All Apache 2.0 licensed.

### Quick Features

- **30-second setup**: Interactive `glassalpha init` wizard
- **Smart defaults**: Auto-detects config files, infers output paths
- **Built-in datasets**: German Credit and Adult Income for quick testing
- **Self-diagnosable errors**: Clear What/Why/Fix error messages
- **Automation support**: `--json-errors` flag for CI/CD pipelines

## CI/CD Integration

GlassAlpha is designed for automation with standardized exit codes and JSON error output:

```bash
# Get machine-readable errors for CI/CD
glassalpha --json-errors audit --config audit.yaml

# Exit codes for scripting
# 0 = Success
# 1 = User error (bad config, missing files)
# 2 = System error (permissions, resources)
# 3 = Validation error (compliance failures)
```

**Auto-detection**: JSON errors automatically enable in GitHub Actions, GitLab CI, CircleCI, Jenkins, and Travis.

**Environment variable**: Set `GLASSALPHA_JSON_ERRORS=1` to enable JSON output.

Example JSON error output:

```json
{
  "status": "error",
  "exit_code": 1,
  "error": {
    "type": "CONFIG",
    "message": "File 'config.yaml' does not exist",
    "details": {},
    "context": { "config_path": "config.yaml" }
  },
  "timestamp": "2025-10-05T12:00:00Z"
}
```

## Learn more

- **[Documentation](https://glassalpha.com/)** - User guides, API reference, and tutorials
- **[Developer guide](packages/README.md)** - Architecture deep-dive and contribution guide
- **[German credit tutorial](https://glassalpha.com/examples/german-credit-audit/)** - Step-by-step walkthrough with a real dataset
- **[About GlassAlpha](https://glassalpha.com/about/)** - Who, what & why.

## Contributing

I'm a one man band, so quality contributions are welcome.

Found a bug? Want to add a model type? PRs welcome! Check the [contributing guide](https://glassalpha.com/reference/contributing/) for dev setup.

The architecture is designed to be extensible. Adding new models, explainers, or metrics shouldn't require touching core code.

## License

The core library is Apache 2.0. See [LICENSE](LICENSE) for the legal stuff.

Enterprise features/support may be added separately if there's demand for more advanced/custom functionality, but the core will always remain open and free. The name "GlassAlpha" is trademarked to keep things unambiguous. Details in [TRADEMARK.md](TRADEMARK.md).

For dependency licenses and third-party components, check the [detailed licensing info](packages/README.md#license--dependencies).
