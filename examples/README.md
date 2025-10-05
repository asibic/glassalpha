# GlassAlpha Examples

This directory contains runnable Jupyter notebooks demonstrating GlassAlpha features.

## Structure

Each feature has a corresponding notebook:
- `{feature-slug}.ipynb` - Demonstrates the feature with runnable code

## Requirements

Every notebook must:
- Include a smoke cell (version, seed, platform)
- Run end-to-end without manual edits
- Set all random seeds explicitly
- Use repo data or tiny synthetic datasets (no downloads)
- Demonstrate CLI equivalency
- Save at least one deterministic artifact
- Link to API docs in first cell

## Running Examples

```bash
# Install dependencies
pip install glassalpha jupyter

# Launch notebook
jupyter notebook examples/{feature-slug}.ipynb
```

## Validation

Before contributing a new notebook:
1. Run it twice in the same environment
2. Verify artifacts are byte-identical
3. Check all cross-links resolve
4. Keep total cells under 30 (prefer 10-20)

See `.cursor/rules/docs.mdc` for complete notebook standards.

