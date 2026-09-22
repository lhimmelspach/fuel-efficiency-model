# Fuel Efficiency Model

Predicting EPA-style vehicle fuel efficiency from vehicle attributes with a reproducible, leakage-aware regression workflow.

## Executive summary

This repository packages a notebook-first regression project that estimates **combined MPG (`combined_mpg_ft1`)** from vehicle specifications such as make, model, year, engine details, drivetrain, transmission, and fuel type.

The project is designed to show hiring managers that the analysis is more than “I trained a model in a notebook”:

- explicit **target and leakage controls**
- reproducible **random seeds and pipeline-based preprocessing**
- a defensible **grouped validation strategy**
- baseline vs. tree-model **metric comparison**
- interpretable **feature importance and error analysis**

> **Verified-results status:** the repository intentionally does **not** include the external EPA-style dataset, so numeric model results must be generated locally by running the notebook with the documented input file. The notebook is structured to produce those results and save compact artifacts under `artifacts/`.

## Problem statement

Estimate vehicle fuel efficiency from structured vehicle characteristics while avoiding target leakage from direct MPG, cost, emissions, or other post-outcome measurements.

This kind of model is useful for:

- comparing vehicle efficiency across trims and powertrains
- understanding which specifications most influence fuel economy
- illustrating a realistic tabular machine learning workflow with strong validation hygiene

## Dataset

- **Expected source:** EPA-style fuel economy CSV/ZIP downloaded separately by the user
- **Expected scale:** approximately **38,113 rows × 81 columns**
- **Target:** `combined_mpg_ft1`
- **Repository policy:** do **not** commit the external CSV/ZIP dataset or any private data

### How to supply the data

Place the dataset in one of these locations before running the notebook:

1. `data/` (recommended)
2. the repository root
3. Google Colab upload prompt

The notebook automatically searches those locations and can also read a ZIP containing a CSV.

## Methodology

The revised notebook now follows a coherent top-to-bottom workflow:

1. **Locate and load** the external dataset
2. **Coerce numeric-like text fields** safely
3. **Engineer row-level features**, including:
   - vehicle age
   - parsed engine displacement/cylinder hints from engine descriptors
   - transmission gear extraction
   - alternative-powertrain flags
4. **Explicitly exclude non-predictive or dangerous fields**, including:
   - the target itself
   - MPG columns and other target-derived measures
   - post-outcome emissions, cost, range, and charging fields
   - identifier/metadata fields
   - constant or fully missing columns
5. **Audit suspicious target-like columns** using equality and near-equality checks
6. **Fit preprocessing inside scikit-learn pipelines** so imputers and encoders learn only from training folds
7. **Compare models** using consistent regression metrics
8. **Fit the best model on a holdout split** for plots, interpretation, and error analysis

## Validation strategy

For this dataset, random row splits can be too optimistic because many rows represent closely related vehicles from the same make/model family.

### Default strategy

- **Primary:** `GroupKFold` and `GroupShuffleSplit`
- **Grouping key:** `make + model` when available
- **Fallback:** shuffled random K-fold / holdout only if grouped validation is not possible

### Why this is the most credible default here

- It reduces leakage across near-duplicate trims and repeated vehicle families.
- It better reflects the challenge of predicting MPG for unseen vehicle families.
- It is more defensible than a naïve random split for a multi-year structured vehicle table.

### Important limitation

If the real goal is **future model-year forecasting**, a **year-based holdout** should be added as a follow-up experiment. The notebook calls this out explicitly in the limitations section.

## Models compared

The notebook compares:

- **Mean baseline** (`DummyRegressor`)
- **Random Forest**
- **LightGBM** *(optional, if installed)*
- **XGBoost** *(optional, if installed)*

### Metrics reported

- **MAE**
- **RMSE**
- **R²**

## Results

### Cross-validation comparison table

The notebook writes a compact results table to `artifacts/model_comparison.csv`.

| Model | CV MAE | CV RMSE | CV R² | Status |
|---|---:|---:|---:|---|
| Mean baseline | generated at runtime | generated at runtime | generated at runtime | requires local execution |
| Random forest | generated at runtime | generated at runtime | generated at runtime | requires local execution |
| LightGBM* | generated at runtime | generated at runtime | generated at runtime | optional dependency |
| XGBoost* | generated at runtime | generated at runtime | generated at runtime | optional dependency |

\* Included only when the dependency is installed.

### Holdout artifacts

When data is available, the notebook also saves:

- `artifacts/holdout_metrics.json`
- `artifacts/holdout_predictions.csv`
- `artifacts/figures/target_distribution.png`
- `artifacts/figures/holdout_diagnostics.png`
- `artifacts/figures/permutation_importance.png`

## Interpretation and error analysis

The notebook includes:

- **target distribution** visualization
- **predicted-vs-actual** plot
- **residual plot**
- **permutation importance** ranking
- optional **SHAP** analysis when dependencies are available
- **worst-error examples**
- **group-level error summaries** for fields like class, make, and fuel type when present

The analysis is set up to discuss common failure modes such as:

- repeated vehicle families across years
- sparse categories and rare trims
- hybrids and EVs behaving differently from conventional gasoline vehicles
- changing standards or technology across time
- limited ability to extrapolate to future model years without a time-aware split

## Reproducibility

The notebook now includes:

- fixed random seeds for Python/NumPy/model settings
- deterministic train/validation splitting configuration
- preprocessing inside pipelines to avoid fold leakage
- guarded optional explainability steps
- compact artifact output under `artifacts/`

## Setup

### 1) Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Optional extras

Install these only if you want additional models or explainability:

```bash
pip install lightgbm xgboost shap
```

### 3) Add the dataset

```text
data/your_epa_vehicle_file.csv
```

### 4) Run the notebook

Open and run `/home/runner/work/fuel-efficiency-model/fuel-efficiency-model/dmpfinal.ipynb` top to bottom in Jupyter or Google Colab.

## Repository structure

```text
fuel-efficiency-model/
├── dmpfinal.ipynb      # main analysis notebook
├── README.md           # project narrative and setup
├── requirements.txt    # core dependencies (+ optional extras documented in comments)
└── artifacts/          # generated locally at runtime; not committed
```

## What changed in this revision

Compared with the earlier notebook, this version:

- removes duplicate/fragmented exploratory sections in favor of one coherent workflow
- makes target selection explicit and safer
- documents and exports excluded fields plus leakage-audit outputs
- avoids preprocessing leakage by fitting encoders/imputers inside pipelines
- adds a simple baseline and a compact model comparison table
- adds reproducibility controls and cleaner runtime behavior
- adds error-analysis outputs and lightweight saved figures

## Limitations

- Results are only as credible as the supplied external dataset.
- Grouped validation is better than random splitting here, but it is still not a substitute for a true future-year holdout.
- Rare trims, unusual powertrains, and newly introduced technologies may be underrepresented.
- Optional models depend on extra packages that are not required for the core workflow.

## Author

**Luke Himmelspach**

This repository was refined to better communicate technical judgment, modeling rigor, and reproducibility to prospective data science hiring managers while preserving Luke Himmelspach’s authorship.
