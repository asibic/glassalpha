# glassalpha

Interpretability tooling. Repo contains:
- `packages/glassalpha`: Python package
- `site`: MkDocs docs (isolated env)

## Dev quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e packages/glassalpha[dev]
pytest
cd site && pip install -r requirements.txt && mkdocs serve
