# CGMacros — Postprandial Glycemia Prediction

> Predicting blood glucose response after a meal using continuous glucose monitoring (CGM), meal macronutrients, physical activity, and clinical profiles — across healthy, prediabetic, and type 2 diabetic participants.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Data](https://img.shields.io/badge/Data-PhysioNet-lightgrey)

---

## Overview

This project addresses postprandial glycemia prediction through two complementary tasks:

- **Task 1 — Regression**: predict the exact blood glucose value (mg/dL) at t+30, t+60, and t+90 minutes after meal onset
- **Task 2 — Classification**: predict the glycemic risk category (euglycemia / hyperglycemia)

Models span from a linear regression baseline to decision trees and random forests, all evaluated via GroupKFold cross-validation on a real-world clinical dataset of 44 participants with diverse metabolic profiles.

---

## Dataset — CGMacros

**Source**: [PhysioNet CGMacros v1.0.0](https://physionet.org/content/cgmacros/1.0.0/)  
**Access**: requires CITI Program certification + PhysioNet Data Use Agreement — see [`data/README.md`](data/README.md)

| Parameter | Value |
| :--- | :--- |
| Participants | 44 (45 enrolled; patient #12 excluded — cardiac comorbidity) |
| Groups | 15 healthy / 16 prediabetic / 14 type 2 diabetic |
| Follow-up | 10 days under real-life conditions |
| CGM sensor retained | Abbott FreeStyle Libre Pro (15 min → interpolated to 1 min) |
| CGM sensor excluded | Dexcom G6 Pro (significant data gaps and inconsistencies) |
| Raw glucose points | ~129,600 |
| Extracted meal windows | ~1,700 |

> ⚠️ This repository contains **no patient data**. All raw and processed files are excluded via `.gitignore` in compliance with PhysioNet Terms of Use and GDPR.

---

## Results

*All results are based on GroupKFold cross-validation (k=5, grouped by patient ID to prevent data leakage). Values reported as mean ± std.*

### Task 1 — Regression

| Model | RMSE t+30 | RMSE t+60 | RMSE t+90 | MAE t+60 | R² t+60 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| OLS (baseline) | 15.1 ± 0.7 | 30.4 ± 2.9 | 31.9 ± 3.2 | 23.5 ± 2.5 | 0.52 ± 0.11 |
| Ridge | 15.2 ± 0.9 | 30.5 ± 3.0 | 31.9 ± 3.2 | 23.6 ± 2.5 | 0.52 ± 0.11 |
| Lasso | 15.2 ± 0.9 | 30.6 ± 3.3 | 31.5 ± 3.5 | 23.5 ± 2.5 | 0.51 ± 0.12 |
| Lasso → RF | 19.0 ± 2.7 | 31.3 ± 3.9 | 31.0 ± 5.4 | 23.8 ± 2.7 | 0.49 ± 0.13 |
| Decision Tree | 20.9 ± 0.8 | 36.3 ± 2.2 | 36.6 ± 4.9 | 27.0 ± 1.6 | 0.32 ± 0.28 |
| **Random Forest** | 19.4 ± 1.4 | **31.0 ± 2.5** | 31.9 ± 2.6 | **23.5 ± 2.2** | **0.52 ± 0.15** |

Units: RMSE and MAE in mg/dL. Clinical acceptability threshold (ISO 15197): 15 mg/dL — achieved by linear models at t+30 min only.

<p align="center">
  <img src="results/figures/compare_rmse.png" width="48%"/>
  <img src="results/figures/radar_t60.png" width="48%"/>
</p>

### Task 2 — Classification (hypoglycemia / euglycemia / hyperglycemia)

| Model | Accuracy t+30 | Accuracy t+60 | F1-macro t+30 | F1-macro t+60 | F1-macro t+90 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.879 ± 0.024** | 0.799 ± 0.051 | **0.835 ± 0.040** | **0.773 ± 0.048** | 0.759 ± 0.055 |
| Decision Tree | 0.844 ± 0.052 | 0.778 ± 0.047 | 0.789 ± 0.074 | 0.729 ± 0.050 | 0.755 ± 0.068 |
| **Random Forest** | 0.882 ± 0.028 | **0.817 ± 0.057** | 0.829 ± 0.047 | 0.781 ± 0.057 | **0.781 ± 0.067** |

Recall macro at t+60 min: Logistic Regression **0.800**, Random Forest **0.791**, Decision Tree **0.733**.  
Recall is prioritised over accuracy given the clinical cost of false negatives in hyperglycemia detection.

<p align="center">
  <img src="results/figures/heatmap_f1.png" width="60%"/>
</p>

---

## Repository Structure

```
cgmacros-glycemia-prediction/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── README.md                                   ← PhysioNet + CITI access instructions
│
├── src/
│   ├── config.py
│   ├── build_patient_table.py
│   ├── meal_window_builder.py
│   ├── column_description_meal_window.py
│   ├── baseline_linear_regression.py
│   ├── task1_linear_models.py
│   ├── task1_decision_tree.py
│   ├── task1_random_forest.py
│   ├── compare_task1_regression.py
│   ├── task2_logistic_regression.py
│   ├── task2_trees_classification.py
│   └── compare_task2_classification.py
│
├── notebooks/
│   └── EDA_CGMacros.ipynb
│
├── results/
│   │
│   ├── baseline_linear/                            ← baseline_linear_regression.py
│   │   ├── baseline_results.csv
│   │   ├── coefficients_t30.png
│   │   ├── coefficients_t60.png
│   │   ├── coefficients_t90.png
│   │   ├── scatter_t30.png
│   │   ├── scatter_t60.png
│   │   └── scatter_t90.png
│   │
│   ├── task1_linear_models/                        ← task1_linear_models.py
│   │   ├── results_linear_models.csv
│   │   ├── coef_table_t30.csv
│   │   ├── coef_table_t60.csv
│   │   ├── coef_table_t90.csv
│   │   ├── lasso_features_t30.csv
│   │   ├── lasso_features_t60.csv
│   │   ├── lasso_features_t90.csv
│   │   ├── coefficients_comparison_t30.png
│   │   ├── coefficients_comparison_t60.png
│   │   ├── coefficients_comparison_t90.png
│   │   ├── importance_LassoRF_t30.png
│   │   ├── importance_LassoRF_t60.png
│   │   ├── importance_LassoRF_t90.png
│   │   ├── lasso_alpha_t30.png
│   │   ├── lasso_alpha_t60.png
│   │   ├── lasso_alpha_t90.png
│   │   ├── residuals_OLS_t30.png
│   │   ├── residuals_OLS_t60.png
│   │   ├── residuals_OLS_t90.png
│   │   ├── scatter_Lasso_t30.png
│   │   ├── scatter_Lasso_t60.png
│   │   ├── scatter_Lasso_t90.png
│   │   ├── scatter_LassoRF_t30.png
│   │   ├── scatter_LassoRF_t60.png
│   │   ├── scatter_LassoRF_t90.png
│   │   ├── scatter_OLS_t30.png
│   │   ├── scatter_OLS_t60.png
│   │   ├── scatter_OLS_t90.png
│   │   ├── scatter_Ridge_t30.png
│   │   ├── scatter_Ridge_t60.png
│   │   └── scatter_Ridge_t90.png
│   │
│   ├── task1_decision_tree/                        ← task1_decision_tree.py
│   │   ├── results_decision_tree_regression.csv
│   │   ├── importance_t30.png
│   │   ├── importance_t60.png
│   │   ├── importance_t90.png
│   │   ├── scatter_t30.png
│   │   ├── scatter_t60.png
│   │   ├── scatter_t90.png
│   │   ├── tree_rules_t60.txt
│   │   └── tree_structure_t60.png
│   │
│   ├── task1_random_forest/                        ← task1_random_forest.py
│   │   ├── results_random_forest_regression.csv
│   │   ├── gridsearch_best_params_t60.csv
│   │   ├── importance_t30.png
│   │   ├── importance_t60.png
│   │   ├── importance_t90.png
│   │   ├── learning_curve_t60.png
│   │   ├── scatter_t30.png
│   │   ├── scatter_t60.png
│   │   └── scatter_t90.png
│   │
│   ├── comparison_task1_regression/                ← compare_task1_regression.py
│   │   ├── comparison_regression_summary.csv
│   │   ├── compare_mae.png
│   │   ├── compare_r2.png
│   │   ├── compare_rmse.png
│   │   └── radar_t60.png
│   │
│   ├── task2_logistic_regression/                  ← task2_logistic_regression.py
│   │   ├── results_logistic_regression_classification.csv
│   │   ├── classification_report_t30.csv
│   │   ├── classification_report_t60.csv
│   │   ├── classification_report_t90.csv
│   │   ├── confusion_matrix_t30.png
│   │   ├── confusion_matrix_t60.png
│   │   ├── confusion_matrix_t90.png
│   │   ├── metrics_by_class_t30.png
│   │   ├── metrics_by_class_t60.png
│   │   ├── metrics_by_class_t90.png
│   │   ├── roc_curves_t30.png
│   │   ├── roc_curves_t60.png
│   │   └── roc_curves_t90.png
│   │
│   ├── task2_decision_tree/                        ← task2_trees_classification.py
│   │   ├── results_decision_tree_classification.csv
│   │   ├── confusion_t30.png
│   │   ├── confusion_t60.png
│   │   ├── confusion_t90.png
│   │   ├── importance_t30.png
│   │   ├── importance_t60.png
│   │   ├── importance_t90.png
│   │   ├── report_t30.csv
│   │   ├── report_t60.csv
│   │   ├── report_t90.csv
│   │   └── tree_structure_t60.png
│   │
│   ├── task2_random_forest/                        ← task2_trees_classification.py
│   │   ├── results_random_forest_classification.csv
│   │   ├── confusion_t30.png
│   │   ├── confusion_t60.png
│   │   ├── confusion_t90.png
│   │   ├── importance_t30.png
│   │   ├── importance_t60.png
│   │   ├── importance_t90.png
│   │   ├── report_t30.csv
│   │   ├── report_t60.csv
│   │   └── report_t90.csv
│   │
│   └── comparison_task2_classification/            ← compare_task2_classification.py
│       ├── comparison_classification_summary.csv
│       ├── compare_accuracy.png
│       ├── compare_f1.png
│       ├── compare_recall.png
│       ├── focus_recall_t60.png
│       └── heatmap_f1.png
│
└── report/
    └── rapport_CGMacros.pdf
'''

---

## Installation & Usage

**1. Clone the repository**
```bash
git clone https://github.com/BounyMathieu/cgmacros-glycemia-prediction.git
cd cgmacros-glycemia-prediction
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set up the data**

See [`data/README.md`](data/README.md) for step-by-step PhysioNet access and CITI certification instructions. Place the raw files in `data/raw/`.

**4. Run the pipeline**
```bash
# Build patient feature table
python src/build_patient_table.py

# Extract meal windows (~1,700 episodes)
python src/meal_window_builder.py

# Task 1 — Regression
python src/task1_random_forest.py
python src/compare_task1_regression.py

# Task 2 — Classification
python src/task2_logistic_regression.py
python src/compare_task2_classification.py
```

---

## Methods

### Data pipeline

```
bio.csv + CGMacros-0XX.xlsx
        ↓
Add 'group' column (derived from HbA1c thresholds)
Remove patient #12 (cardiac comorbidity unrelated to glycemia)
Drop Dexcom G6 signal (data gaps > threshold)
        ↓
meal_window_builder.py
→ ~1,700 meal windows
→ Filter: minimum 30 valid CGM points in the 60 min pre-meal window
        ↓
GroupKFold cross-validation, k=5 (groups = patient_id)
        ↓
Task 1: Regression    |    Task 2: Classification
```

### Key methodological choices

- **Sensor selection**: Abbott FreeStyle Libre Pro was retained over Dexcom G6 due to fewer data gaps and better temporal consistency across patients.
- **Patient exclusion**: patient #12 was excluded due to abnormally elevated triglycerides and LDL attributed to an independent cardiac condition unrelated to glycemic dysregulation.
- **Cross-validation**: GroupKFold (k=5) stratified by patient ID ensures no patient appears in both train and test sets, preventing data leakage and producing realistic generalization estimates.
- **Class imbalance**: handled via `class_weight='balanced'` in all classification models.
- **Demographic bias**: the cohort includes approximately 2× more female than male participants and is predominantly Hispanic/Latino — results should be interpreted accordingly.

---

## Authors

| Name | Email |
| :--- | :--- |
| Raphaël Pallière | palliere.raphael@icloud.com |
| Mathieu Bouny | bouny.mathieu@gmail.com |

*Master's project — CITI Program accreditation for human subjects research.*

---

## License

This project is licensed under the MIT License.  
The CGMacros dataset is subject to PhysioNet Terms of Use — see [physionet.org](https://physionet.org/content/cgmacros/1.0.0/).
