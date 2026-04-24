"""
Régression continue : prédiction de la glycémie à t+30, t+60, t+90 min.
Modèle    : Arbre de décision (CART) — DecisionTreeRegressor
L'arbre de décision est le deuxième modèle testé après la régression linéaire (baseline).
Son intérêt principal dans ce projet est son interprétabilité native : on peut visualiser
l'arbre et comprendre exactement quelle règle clinique le modèle a apprise.

Validation : GroupKFold k=5 strictement par patient (pas de data leakage)
Config : sélection des features sélectionné dans A : agrégats CGM pré-repas uniquement

Auteurs : Palliere Raphael
"""


import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from config import (
    DATASET_PATH, RESULTS_DIR, RANDOM_STATE, N_FOLDS,
    CLINICAL_RMSE_THRESHOLD, REGRESSION_TARGETS,
    load_dataset, build_X, get_preprocessing, get_cv_splits, save_results,
)


warnings.filterwarnings("ignore")
 
OUTPUT_DIR = os.path.join(RESULTS_DIR, "task1_decision_tree")
MODEL_NAME = "DecisionTree"

#Hyperparamètres de l'arbre
# max_depth limité volontairement pour éviter le surapprentissage sur 45 patients
# et conserver l'interprétabilité (un arbre profond de 60 niveaux n'est pas lisible)
TREE_PARAMS = {
    "max_depth": 5,
    "min_samples_leaf":10,   # Au moins 10 fenêtres repas par feuille
    "random_state": RANDOM_STATE,
}



#Evaluation par Fold CV strictement par patient, avec calcul de RMSE, MAE, R² et interprétation clinique
def evaluate_fold(pipeline, X_train, y_train, X_test, y_test) -> dict:
    """Entraîne et évalue le pipeline sur un fold. Retourne les métriques."""
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
 
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae":float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
        "y_test": y_test.values,
        "y_pred": y_pred,
    }


#Visualisation de l'arbre de décision
def plot_decision_tree(pipeline: Pipeline, feature_names: list, horizon: str, output_path: str):
    tree = pipeline.named_steps["model"]
    fig, ax = plt.subplots(figsize=(25, 10))
    plot_tree(
        tree,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        max_depth=5,           
        ax=ax,
        impurity=False,
        precision=1,
    )
    ax.set_title(
        f"Arbre de décision — Régression glycémie {horizon}\n"
        f"(max_depth={TREE_PARAMS['max_depth']}, visualisation 6 niveaux)",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  → Arbre visualisé : {output_path}")


def plot_scatter(y_test, y_pred, horizon: str, rmse: float, mae: float, output_path: str):
    """Scatter prédit vs réel avec seuils cliniques."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, alpha=0.55, color="#1D9E75", s=35)
 
    lims = [min(y_test.min(), y_pred.min()) - 5, max(y_test.max(), y_pred.max()) + 5]
    ax.plot(lims, lims, "k--", linewidth=1, label="Prédiction parfaite")
    ax.fill_between(lims, [l - 15 for l in lims], [l + 15 for l in lims], alpha=0.1, color="#1D9E75", label="±15 mg/dL")
 
    for seuil, label, color in [(70, "Hypo<70", "#E24B4A"), (140, "Hyper>140", "#EF9F27")]:
        ax.axhline(seuil, color=color, linestyle=":", linewidth=0.8, alpha=0.7)
        ax.text(lims[0] + 1, seuil + 1, label, fontsize=8, color=color)
 
    ax.set_xlabel("Glycémie réelle (mg/dL)", fontsize=10)
    ax.set_ylabel("Glycémie prédite (mg/dL)", fontsize=10)
    ax.set_title(f"Arbre de décision — {horizon}\nRMSE={rmse:.1f} | MAE={mae:.1f} mg/dL", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_xlim(lims); ax.set_ylim(lims)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()





 


def plot_feature_importance(pipeline: Pipeline, feature_names: list, horizon: str, output_path: str):
    """Barplot de l'importance des features par réduction d'impureté (Gini)."""
    importances = pipeline.named_steps["model"].feature_importances_
    idx = np.argsort(importances)[::-1][:20]
 
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(
        [feature_names[i] for i in idx][::-1],
        importances[idx][::-1],
        color="#1D9E75", alpha=0.85,
    )
    ax.set_xlabel("Importance (réduction d'impureté)", fontsize=10)
    ax.set_title(f"Importance des features — Arbre de décision {horizon}", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" -> Importance features : {output_path}")







#Pipeline de modélisation : préprocessing + arbre de décision
def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
 
    print(f"TÂCHE 1 — Régression | Modèle : {MODEL_NAME}")
 
    df= load_dataset(DATASET_PATH)
    X  = build_X(df)
    groups = df["patient_id"]
 
    print(f"\n Dataset : {len(df)} fenêtres | {df['patient_id'].nunique()} patients")
    print(f"Features : {X.shape[1]} colonnes (Config A)")
 
    all_results = []
 
    for horizon_key, target_col in REGRESSION_TARGETS.items():
        y = df[target_col].copy()
        valid = y.notna()
        X_v, y_v, g_v = X[valid].reset_index(drop=True), y[valid].reset_index(drop=True), groups[valid].reset_index(drop=True)
        print(f"\n  Horizon {horizon_key} ({target_col}) — {valid.sum()} fenêtres valides")
 
        pipeline = Pipeline(
            get_preprocessing() + [("model", DecisionTreeRegressor(**TREE_PARAMS))]
        )
        splits  = get_cv_splits(X_v, y_v, g_v)
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
 
            print(f"Fold {fold_idx+1} | RMSE={fold_result['rmse']:.2f} | "
                  f"MAE={fold_result['mae']:.2f} | R²={fold_result['r2']:.3f}")
 
        rmse_m, rmse_s = np.mean(rmses), np.std(rmses)
        mae_m,  mae_s= np.mean(maes),  np.std(maes)
        r2_m,   r2_s = np.mean(r2s),   np.std(r2s)
 
        print(f"\n -> Moyenne | RMSE={rmse_m:.2f}±{rmse_s:.2f} | "
              f"MAE={mae_m:.2f}±{mae_s:.2f} | R²={r2_m:.3f}±{r2_s:.3f}")
        
        # Graphiques sur le dernier fold
        feature_names = list(X_v.columns)
 
        plot_scatter(
            last_y_test, last_y_pred, f"t+{horizon_key[1:]} min",
            rmse_m, mae_m,
            output_path=os.path.join(OUTPUT_DIR, f"scatter_{horizon_key}.png"),
        )
        plot_feature_importance(
            pipeline, feature_names, f"t+{horizon_key[1:]} min",
            output_path=os.path.join(OUTPUT_DIR, f"importance_{horizon_key}.png"),
        )
 
        # Visualiser l'arbre uniquement pour t+60 seuleent
        if horizon_key == "t60":
            plot_decision_tree(
                pipeline, feature_names, "t+60 min",
                output_path=os.path.join(OUTPUT_DIR, "tree_structure_t60.png"),
            )
            # Export texte de l'arbre pour une analyse détaillée
            tree_text = export_text(
                pipeline.named_steps["model"],
                feature_names=feature_names,
                max_depth=10,
            )
            with open(os.path.join(OUTPUT_DIR, "tree_rules_t60.txt"), "w") as f:
                f.write(tree_text)
            print(f" -> Règles arbre exportées : {OUTPUT_DIR}/tree_rules_t60.txt")
 
        all_results.append({
            "model": MODEL_NAME,
            "task":"regression",
            "horizon":   f"t+{horizon_key[1:]} min",
            "rmse_mean":round(rmse_m, 2),
            "rmse_std": round(rmse_s, 2),
            "mae_mean": round(mae_m, 2),
            "mae_std": round(mae_s, 2),
            "r2_mean":  round(r2_m, 3),
            "r2_std":round(r2_s, 3),
            "n_features": X_v.shape[1],
            "max_depth": TREE_PARAMS["max_depth"],
            "clinical_ok": rmse_m <= CLINICAL_RMSE_THRESHOLD,
        })
 
    save_results(all_results, os.path.join(OUTPUT_DIR, "results_decision_tree_regression.csv"))
 
 
if __name__ == "__main__":
    run()
        






