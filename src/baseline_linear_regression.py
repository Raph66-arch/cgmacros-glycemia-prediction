"""
baseline_linear_regression.py

Régression continue : prédiction de la glycémie à t+30, t+60, t+90 min.


Validation : GroupKFold k=5 strictement par patient
Config     : A — agrégats CGM pré-repas uniquement
 
Auteurs : Palliere Raphael & Bouny Mathieu — E4 Bio
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
 
from config import (
    DATASET_PATH, RESULTS_DIR, RANDOM_STATE,
    CLINICAL_RMSE_THRESHOLD, REGRESSION_TARGETS,
    load_dataset, build_X, get_preprocessing, get_cv_splits, save_results,
)
 
warnings.filterwarnings("ignore")
 
OUTPUT_DIR = os.path.join(RESULTS_DIR, "baseline_linear")
MODEL_NAME = "Ridge"
 
RIDGE_PARAMS = {
    "alpha": 1.0,
    "random_state": RANDOM_STATE,
}
 


#Evaluation par Fold CV strictement par patient, avec calcul de RMSE, MAE, R² et interprétation clinique
def evaluate_fold(pipeline, X_train, y_train, X_test, y_test) -> dict:
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    return {
        "rmse":   float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae":    float(mean_absolute_error(y_test, y_pred)),
        "r2":     float(r2_score(y_test, y_pred)),
        "y_test": y_test.values,
        "y_pred": y_pred,
    }


#Visualisation 
def plot_scatter(y_test, y_pred, horizon: str, rmse: float, mae: float, output_path: str):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, alpha=0.55, color="#888780", s=35)
    lims = [min(y_test.min(), y_pred.min()) - 5, max(y_test.max(), y_pred.max()) + 5]
    ax.plot(lims, lims, "k--", linewidth=1, label="Prédiction parfaite")
    ax.fill_between(lims, [l - 15 for l in lims], [l + 15 for l in lims], alpha=0.12, color="#888780", label="±15 mg/dL")
    for seuil, label, color in [(70, "Hypo<70", "#E24B4A"), (140, "Hyper>140", "#EF9F27")]:
        ax.axhline(seuil, color=color, linestyle=":", linewidth=0.8, alpha=0.7)
        ax.text(lims[0] + 1, seuil + 1, label, fontsize=8, color=color)
    ax.set_xlabel("Glycémie réelle (mg/dL)", fontsize=10)
    ax.set_ylabel("Glycémie prédite (mg/dL)", fontsize=10)
    ax.set_title(f"Ridge (baseline)—{horizon}\nRMSE={rmse:.1f} | MAE={mae:.1f} mg/dL", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_xlim(lims); ax.set_ylim(lims)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" -> Scatter : {output_path}")
 


def plot_coefficients(pipeline: Pipeline, feature_names: list, horizon: str, output_path: str):
    coefs = pipeline.named_steps["model"].coef_
    df_coef = pd.DataFrame({
        "feature":feature_names,
        "coefficient": coefs,
        "abs_coef":  np.abs(coefs),
    }).sort_values("abs_coef", ascending=False).head(20)
    colors = ["#1D9E75" if c >= 0 else "#E24B4A" for c in df_coef["coefficient"]]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(df_coef["feature"][::-1], df_coef["coefficient"][::-1], color=colors[::-1], alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient (après normalisation)", fontsize=10)
    ax.set_title(f"Ridge — Coefficients | {horizon}", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" -> Coefficients : {output_path}")
 

#Pipeline 
def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"TACHE 1 — Régression | Modèle : {MODEL_NAME} (baseline)")
 
    df= load_dataset(DATASET_PATH)
    X = build_X(df)
    groups= df["patient_id"]
 
    print(f"\n Dataset : {len(df)} fenêtres | {df['patient_id'].nunique()} patients")
    print(f"Features : {X.shape[1]} colonnes (Config A)")
 
    all_results = []
 
    for horizon_key, target_col in REGRESSION_TARGETS.items():
        y = df[target_col].copy()
        valid = y.notna()
        X_v = X[valid].reset_index(drop=True)
        y_v = y[valid].reset_index(drop=True)
        g_v = groups[valid].reset_index(drop=True)
 
        print(f"\n Horizon {horizon_key} ({target_col}) — {valid.sum()} fenêtres valides")
 
        pipeline = Pipeline(
            get_preprocessing() + [("model", Ridge(**RIDGE_PARAMS))]
        )
        splits = get_cv_splits(X_v, y_v, g_v)
        rmses, maes, r2s = [], [], []
        last_y_test, last_y_pred = None, None
 
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            fold_result = evaluate_fold(
                pipeline,
                X_v.iloc[train_idx], y_v.iloc[train_idx],
                X_v.iloc[test_idx],  y_v.iloc[test_idx],
            )
            rmses.append(fold_result["rmse"])
            maes.append(fold_result["mae"])
            r2s.append(fold_result["r2"])
            last_y_test = fold_result["y_test"]
            last_y_pred = fold_result["y_pred"]
 
            print(f"    Fold {fold_idx+1} | RMSE={fold_result['rmse']:.2f} | "f"MAE={fold_result['mae']:.2f} | R²={fold_result['r2']:.3f}")
 
        rmse_m, rmse_s= np.mean(rmses), np.std(rmses)
        mae_m,mae_s = np.mean(maes),  np.std(maes)
        r2_m,r2_s= np.mean(r2s),   np.std(r2s)
 
        print(f"\n -> Moyenne | RMSE={rmse_m:.2f}±{rmse_s:.2f} | "
              f"MAE={mae_m:.2f}±{mae_s:.2f} | R²={r2_m:.3f}±{r2_s:.3f}")
 
        feature_names= list(X_v.columns)
        horizon_label= f"t+{horizon_key[1:]} min"
 
        plot_scatter(
            last_y_test, last_y_pred, horizon_label, rmse_m, mae_m,
            output_path=os.path.join(OUTPUT_DIR, f"scatter_{horizon_key}.png"),
        )
        plot_coefficients(
            pipeline, feature_names, horizon_label,
            output_path=os.path.join(OUTPUT_DIR, f"coefficients_{horizon_key}.png"),
        )
 
        # Format identique à task1_decision_tree.py et task1_random_forest.py
        all_results.append({
            "model":MODEL_NAME,
            "task": "regression",
            "horizon": horizon_label,
            "rmse_mean":round(rmse_m, 2),
            "rmse_std": round(rmse_s, 2),
            "mae_mean": round(mae_m, 2),
            "mae_std": round(mae_s, 2),
            "r2_mean": round(r2_m, 3),
            "r2_std": round(r2_s, 3),
            "n_features": X_v.shape[1],
            "alpha":RIDGE_PARAMS["alpha"],
            "clinical_ok":rmse_m <= CLINICAL_RMSE_THRESHOLD,
        })
 
    save_results(all_results, os.path.join(OUTPUT_DIR, "baseline_results.csv"))
 
    print("  RÉSUMÉ BASELINE RIDGE")
    for r in all_results:
        ok = "✅" if r["clinical_ok"] else "⚠️ "
        print(f"  {r['horizon']} | RMSE={r['rmse_mean']}±{r['rmse_std']} | "
              f"MAE={r['mae_mean']}±{r['mae_std']} | R²={r['r2_mean']} {ok}")
 
 
if __name__ == "__main__":
    run()


