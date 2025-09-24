# GlassAlpha - Phase 2 Implementation Prompt

## Context

You are continuing the GlassAlpha project, an open-source AI compliance toolkit for ML model auditing. **Phase 1 is 100% complete** with all ML components implemented:

- **5 Model Wrappers**: XGBoost, LightGBM, LogisticRegression, SklearnGeneric, PassThrough
- **3 Explainers**: TreeSHAP (priority 100), KernelSHAP (priority 50), NoOp (fallback)
- **17 Metrics**: 6 Performance, 4 Fairness, 5 Drift, 2 Testing
- **Architecture Foundation**: Registry system, interfaces, config, CLI structure

## Your Task: Phase 2 - Integration & Report Generation

Implement the integration layer that connects all Phase 1 components into a working audit pipeline that generates professional PDF reports.

## Reference Documents (Read These First)

1. **HANDOFF.md** - Complete status of Phase 1 and detailed Phase 2 tasks
2. **ML_COMPONENTS_STATUS.md** - Details of all implemented ML components
3. **NEXT_STEPS.md** - Templates and implementation order for Phase 2
4. **PACKAGE_STRUCTURE.md** - Code organization and architecture patterns

## Phase 2 Goal

Make this command work and generate a professional PDF audit report:
```bash
glassalpha audit --config configs/german_credit.yaml --out audit.pdf --strict
```

## Implementation Checklist

### Week 1: Data & Utilities
- [ ] **Fix Config Loader Bug** (`src/glassalpha/config/loader.py:160`)
  - Fix `apply_profile_defaults()` function call
  - This blocks everything else

- [ ] **Data Module** (`src/glassalpha/data/tabular.py`)
  - CSV/Parquet loading with pandas
  - Schema validation using Pydantic
  - Protected attributes extraction
  - Dataset hashing for reproducibility
  - Train/test split management

- [ ] **Core Utilities**
  - `src/glassalpha/utils/seeds.py` - Centralized random seed management
  - `src/glassalpha/utils/hashing.py` - Deterministic hashing (config, data, model)
  - `src/glassalpha/utils/manifest.py` - Audit manifest with complete lineage

### Week 2: Pipeline Integration
- [ ] **Audit Pipeline** (`src/glassalpha/pipeline/audit.py`)
  - Load data using data module
  - Train/load model (use existing wrappers)
  - Select best explainer automatically (use registry priority)
  - Compute all metrics (performance, fairness, drift)
  - Error handling and progress tracking
  - Generate audit results dictionary

- [ ] **German Credit Dataset** (`src/glassalpha/datasets/german_credit.py`)
  - Download/load German Credit dataset
  - Define schema with sensitive features (age, gender)
  - Create example config in `configs/german_credit.yaml`

### Week 3: Report Generation
- [ ] **HTML Template** (`src/glassalpha/report/templates/standard_audit.html`)
  - Executive summary section
  - Model performance metrics table
  - SHAP explanations (global feature importance)
  - Fairness analysis with demographic breakdowns
  - Drift detection results
  - Audit manifest and reproducibility info

- [ ] **PDF Renderer** (`src/glassalpha/report/renderers/pdf.py`)
  - Use WeasyPrint to convert HTML to PDF
  - Ensure deterministic rendering (fixed timestamps)
  - Include all plots as embedded images

- [ ] **Plotting Module** (`src/glassalpha/report/plots.py`)
  - SHAP summary plots with matplotlib
  - Fairness comparison charts
  - Drift visualization
  - Use fixed random seeds for layout

### Week 4: Testing & Documentation
- [ ] **End-to-End Test**
  - Complete audit on German Credit dataset
  - Verify PDF generation
  - Test reproducibility (same input = identical PDF)
  - Performance target: < 60 seconds

- [ ] **Documentation**
  - "Hello Audit" 5-minute tutorial
  - Update README with usage examples
  - Document all new modules

## Technical Requirements

1. **Determinism**: Every operation must be reproducible with seeds
2. **No Network Calls**: Everything works offline/on-prem
3. **Performance**: Complete audit in < 60 seconds
4. **Professional Output**: PDF suitable for regulatory submission

## Success Criteria

The following command should produce a professional PDF in under 60 seconds:
```bash
glassalpha audit --config configs/german_credit.yaml --out audit.pdf --strict
```

The PDF must include:
- Executive summary with risk assessment
- Performance metrics (all 6 types)
- SHAP explanations (global and local)
- Fairness analysis (all 4 metrics)
- Drift detection results (if applicable)
- Complete audit manifest with hashes

## Architecture Patterns to Follow

1. **Use Existing Components**: All ML components are ready in Phase 1
2. **Registry Pattern**: Use `ModelRegistry.get()`, `ExplainerRegistry.list_by_priority()`
3. **Config-Driven**: Everything configured through YAML
4. **Deterministic**: Use seeds from `utils/seeds.py` everywhere
5. **Error Handling**: Graceful failures with informative messages

## Getting Started

```bash
# 1. Navigate to project
cd /Users/gabe/Sites/glassalpha/packages

# 2. Activate environment
source venv/bin/activate

# 3. Verify Phase 1 components work
python3 -c "
import sys
sys.path.insert(0, 'src')
from glassalpha.core import list_components
print('Components ready:', len(list_components()['models']) +
      len(list_components()['explainers']) + len(list_components()['metrics']))
"
# Should print: Components ready: 25

# 4. Start with fixing the config loader bug
# Then implement data module
# Then build pipeline to connect everything
```

## Notes

- All ML components (models, explainers, metrics) are complete and working
- Focus on integration and report generation only
- The architecture is proven - follow existing patterns
- Test incrementally - get data loading working first
- German Credit dataset is the primary test case

---

**Phase 1 is complete. Time to build the integration layer and generate those PDFs!**
