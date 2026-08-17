Fuel Efficiency Model

A machine learning project for predicting vehicle fuel efficiency using historical EPA data.

Overview:
This project builds predictive models for vehicle fuel efficiency (MPG) using a dataset of 38,113 vehicles spanning multiple decades. The notebook performs comprehensive data analysis, feature engineering, model training, and interpretation.

Dataset
Source: EPA fuel efficiency data
Size: 38,113 vehicles × 81 features
Target Variable: combined_mpg_ft1 (Combined fuel efficiency in MPG)
Key Features:
Vehicle characteristics: make, model, year, class, drive type
Engine specs: displacement, cylinders, descriptor
Transmission information
Fuel type
Features & Capabilities
Data Processing
Auto-detection of target variable from common naming conventions
Numeric coercion for object-type columns containing numeric values
Automatic feature derivation:
Vehicle age (calculated from year)
Engine specifications parsed from descriptors (displacement, cylinders, electrification)
Categorical lumping for high-cardinality variables (top-k with "OTHER" category)
Robust handling of duplicate column labels
Modeling
Multiple algorithms: Random Forest, LightGBM, XGBoost
Cross-validation with standard K-Fold or GroupKFold support
Hyperparameter tuning via Optuna (optional, configurable)
Feature importance analysis using:
Permutation importance
Mutual information scoring
SHAP explanations (optional)
Configuration
Key toggles in the first cell:

DO_TUNING: Enable/disable hyperparameter optimization
TUNING_TRIALS: Number of Optuna trials (default: 40)
DO_SHAP: Compute SHAP feature explanations
SHAP_SAMPLE_N: Sample size for SHAP (default: 500)
N_JOBS: Parallel jobs for training (default: 4)
Requirements
Core Libraries
pandas, numpy
scikit-learn
matplotlib, seaborn
joblib
Optional Libraries
LightGBM: Gradient boosting alternative
XGBoost: Advanced gradient boosting
Optuna: Hyperparameter tuning
SHAP: Model interpretability
Usage
Upload or place your CSV file in the working directory
The notebook will auto-detect the target variable (or set TARGET manually)
Run cells sequentially to:
Load and explore data
Derive features
Train models
Evaluate and interpret results
Outputs
Results are saved to the artifacts/ directory:

Trained pipelines (joblib format)
Performance metrics and tables
Visualizations and reports
SHAP explanations (if enabled)
Environment
Designed for Google Colab with GPU acceleration (T4), but runs on any Python 3.7+ environment with dependencies installed.
