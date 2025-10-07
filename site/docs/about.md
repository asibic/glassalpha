# About GlassAlpha

## What

GlassAlpha makes **deterministic, regulator-ready PDF audit reports** for tabular ML models. It's an open-source ([Apache 2.0](reference/trust-deployment.md#licensing-dependencies)) toolkit for teams who need reproducible, audit-ready model documentation.

Ever tried explaining your ML model to a regulator? It's painful. They want proof your model isn't biased, documentation that's reproducible, and explanations they can actually verify. Most tools either don't cut it for compliance or lock you into expensive platforms you can't inspect.

GlassAlpha generates deterministic audit PDFs with complete lineage tracking. Same config, same data, same seed = byte-identical PDF every time. Every metric is reproducible, every decision is explainable, and the entire audit trail is verifiable.

**Why open source?** Because compliance tools need to be trustworthy. When a regulator asks "how did you calculate this?", you can point them to the exact code. No black boxes, no vendor lock-in, no trust-me-bro.

## Who

I'm [Gabe Mays](https://gmays.com/about/) and I like to build things.

## Why

A few motivations came together to build GlassAlpha.

I've been building AI products since 2023 when I built one of the first AI products at my company. But I wanted to understand how AI actually works.

So, I [started studying math daily](https://gmays.com/how-im-relearning-math-as-an-adult/) **My streak is at 728 days** as of 9/30/2025. I shared my [500 day update here](https://gmays.com/500-days-of-math/)

I've also been investing in AI companies since 2023. AI is incredibly useful, but I kept noticing some companies struggling to adopt it meaningfully.

A big part of the blocker is AI interpretability and explainability, because you can't deploy what you can't explain (especially in regulated industries).

LLMs were still over my head, so I decided to start with simpler tabular models.

The existing tools were cool, but not like the polished products I was used to building. So, I decided to build my own as a way to learn and hopefully also make something useful to others that filled a gap and accelerated the adoption of AI.

I'm trying to make the project largely self-documenting, so if you see any issues in the documentation or otherwise, PRs welcome!
