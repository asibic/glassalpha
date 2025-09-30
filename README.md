# GlassAlpha

**Generate audit-ready ML compliance reports that regulators will actually trust.**

## What is this?

Ever tried explaining your ML model to a regulator? It's painful. They want proof your model isn't biased, documentation that's reproducible, and explanations they can actually verify. Most tools either don't cut it for compliance or lock you into expensive platforms you can't inspect.

GlassAlpha generates deterministic audit PDFs with complete lineage tracking. Same config, same data, same seed = byte-identical PDF every time. Every metric is reproducible, every decision is explainable, and the entire audit trail is verifiable.

**Why open source?** Because compliance tools need to be trustworthy. When a regulator asks "how did you calculate this?", you can point them to the exact code. No black boxes, no vendor lock-in, no trust-me-bro.

## Why build it?

A few motivations came together. I've been building AI products since 2023 when I built one of the first AI products at my company. But I wanted to understand how AI actually works, so I [started studying math daily](https://gmays.com/how-im-relearning-math-as-an-adult/). My streak is at 728 days as of writing this on 9/30/2025 ([also see my 500 day update](https://gmays.com/500-days-of-math/)).

I've also been investing in AI companies since 2023. AI is incredibly useful, but I kept noticing some companies struggling to adopt it meaningfully. A big part of the blocker is AI interpretability and explainability, because you can't deploy what you can't explain (especially in regulated industries).

LLMs were still way over my head, so I decided to start with simpler tabular models. The existing tools were cool, but not like the polished products I was used to building. So, I decided to build my own as a way to learn and hopefully also make something useful to others that filled a gap in the market and accelerated the adoption of AI.

I'm trying to make the project largely self-documenting, so if you see any issues in the documentation or otherwise, PRs welcome!

## Get started

### Your first audit in 60 seconds

```bash
# Clone and install
git clone https://github.com/GlassAlpha/glassalpha
cd glassalpha/packages
pip install -e .

# Generate an audit PDF (uses included German Credit example)
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

It's all Apache 2.0 licensed.

## Learn more

- **[Documentation](https://glassalpha.com/)** - User guides, API reference, and tutorials
- **[Developer guide](packages/README.md)** - Architecture deep-dive and contribution guide
- **[German credit tutorial](https://glassalpha.com/examples/german-credit-audit/)** - Step-by-step walkthrough with a real dataset

## Contributing

I'm a one man band, so any contributions are welcome.

Found a bug? Want to add a model type? PRs welcome! Check the [contributing guide](https://glassalpha.com/reference/contributing/) for dev setup.

The architecture is designed to be extensible. Adding new models, explainers, or metrics shouldn't require touching core code.

## License

The core library is Apache 2.0. See [LICENSE](LICENSE) for the legal stuff.

Enterprise features/support may be added separately if there's demand for more advanced/custom functionality, but the core will always remain open and free. The name "GlassAlpha" is trademarked to keep things unambiguous. Details in [TRADEMARK.md](TRADEMARK.md).

For dependency licenses and third-party components, check the [detailed licensing info](packages/README.md#license--dependencies).

---

_Built because compliance tools shouldn't be black boxes._
