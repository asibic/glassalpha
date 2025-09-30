# GlassAlpha

An open source ([toolkit](https://glassalpha.com/reference/trust-deployment/#licensing-dependencies)) to generate deterministic, regulator-ready PDF audit reports for tabular ML models. More [about GlassAlpha](https://glassalpha.com/about/).

## Get started

### Run your first audit in 60 seconds

Clone and install

```bash
git clone https://github.com/GlassAlpha/glassalpha
cd glassalpha/packages
pip install -e .
```

Generate an audit PDF (uses included German Credit example)

```bash
glassalpha audit --config configs/german_credit_simple.yaml --output audit.pdf
```

That's it. You now have a complete audit report with model performance, SHAP explanations, and fairness metrics.

**Want more details?** Check the [full installation guide](packages/README.md#installation) or jump to the [German Credit tutorial](https://glassalpha.com/examples/german-credit-audit/) to see what's in the report.

## What's included

- **`packages/`** - The actual Python package ([dev docs here](packages/README.md))
- **`site/`** - User documentation and tutorials. The docs site is at [glassalpha.com](https://glassalpha.com/)
- **`configs/`** - Example audit configs you can copy and modify

## What you get

Right now, GlassAlpha handles:

- **Models**: XGBoost, LightGBM, Logistic Regression (more coming)
- **Explanations**: TreeSHAP feature importance with individual prediction breakdowns
- **Fairness**: Demographic parity, equal opportunity, bias detection
- **Output**: Professional PDFs that are byte-identical on repeat runs
- **Everything runs locally** - your data never leaves your machine

All Apache 2.0 licensed.

## Learn more

- **[Documentation](https://glassalpha.com/)** - User guides, API reference, and tutorials
- **[Developer guide](packages/README.md)** - Architecture deep-dive and contribution guide
- **[German credit tutorial](https://glassalpha.com/examples/german-credit-audit/)** - Step-by-step walkthrough with a real dataset

## Contributing

I'm a one man band, so quality contributions are welcome.

Found a bug? Want to add a model type? PRs welcome! Check the [contributing guide](https://glassalpha.com/reference/contributing/) for dev setup.

The architecture is designed to be extensible. Adding new models, explainers, or metrics shouldn't require touching core code.

## License

The core library is Apache 2.0. See [LICENSE](LICENSE) for the legal stuff.

Enterprise features/support may be added separately if there's demand for more advanced/custom functionality, but the core will always remain open and free. The name "GlassAlpha" is trademarked to keep things unambiguous. Details in [TRADEMARK.md](TRADEMARK.md).

For dependency licenses and third-party components, check the [detailed licensing info](packages/README.md#license--dependencies).

---

_Built because compliance tools shouldn't be black boxes._
