"""
task1_random_forest.py

Tâche 1 — Régression continue : prédiction de la glycémie à t+30, t+60, t+90 min.

Modèle    : Random Forest Regressor
Le Random Forest est un ensemble d'arbres de décision entraînés sur des sous-échantillons
bootstrap des données et des sous-ensembles aléatoires de features (bagging).

Il est généralement plus robuste qu'un arbre seul sur de petits datasets et capture mieux les interactions complexes entre features. Cependant, il est moins interprétable que l'arbre de décision simple.
Auteurs : Palliere Raphael & Bouny Mathieu — E4 Bio

"""


import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
from sklearn.model_selection import GroupKFold
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


from config import (
    DATASET_PATH, RESULTS_DIR, RANDOM_STATE, N_FOLDS,
    CLINICAL_RMSE_THRESHOLD, REGRESSION_TARGETS,
    load_dataset, build_X, get_preprocessing, get_cv_splits, save_results,
)
 
warnings.filterwarnings("ignore")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "task1_random_forest")
MODEL_NAME = "RandomForest"

# Hyperparamètres
# n_estimators=200 : bon compromis variance/temps de calcul sur ce volume de données
# max_depth=8 : limite le surapprentissage sans trop brider le modèle
# min_samples_leaf=5 : chaque feuille doit représenter au moins 5 fenêtres repas

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 8,
    "min_samples_leaf": 5,
    "max_features": "sqrt",   # Standard pour la régression (≈ √n_features)
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


#Evaluation par FOLD 
def evaluate_fold(pipeline, X_train, y_train, X_test, y_test) -> dict:
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    return {
        "rmse":float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2":float(r2_score(y_test, y_pred)),
        "y_test": y_test.values,
        "y_pred": y_pred,
    }
 


#Visualisation de l'importance des features
def plot_feature_importance(pipeline: Pipeline, feature_names: list, horizon: str, output_path: str):
    """
    Barplot des importances par réduction d'impureté (moyenne sur tous les arbres).
    Plus stable que l'importance d'un seul arbre.
    """
    importances = pipeline.named_steps["model"].feature_importances_
    std_imp= np.std([t.feature_importances_ for t in pipeline.named_steps["model"].estimators_], axis=0)
    idx = np.argsort(importances)[::-1][:20]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(
        [feature_names[i] for i in idx][::-1],
        importances[idx][::-1],
        xerr=std_imp[idx][::-1],
        color="#1D9E75", alpha=0.85, capsize=3,
    )
    ax.set_xlabel("Importance moyenne (réduction d'impureté)", fontsize=10)
    ax.set_title(f"Random Forest — Importance des features\nHorizon {horizon} (n_estimators={RF_PARAMS['n_estimators']})", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" => Importance features : {output_path}")
 


def plot_scatter(y_test, y_pred, horizon: str, rmse: float, mae: float, output_path: str):
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
    ax.set_title(f"Random Forest — {horizon}\nRMSE={rmse:.1f} | MAE={mae:.1f} mg/dL", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_xlim(lims); ax.set_ylim(lims)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_learning_curve(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, groups: pd.Series, output_path: str):
    """
    Courbe d'apprentissage : RMSE train vs test en fonction de la taille du train.
    Permet de diagnostiquer le surapprentissage (biais/variance).
    """
    from sklearn.model_selection import learning_curve
    from sklearn.model_selection import GroupKFold
    train_sizes, train_scores, test_scores = learning_curve(
        pipeline, X, y,
        cv= GroupKFold(n_splits=N_FOLDS),
        groups=groups,
        scoring="neg_root_mean_squared_error",
        train_sizes=np.linspace(0.2, 1.0, 8),
        n_jobs=-1,
    )
 
    train_rmse = -train_scores.mean(axis=1)
    test_rmse  = -test_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    test_std   = test_scores.std(axis=1)
 
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_sizes, train_rmse, "o-", color="#1D9E75", label="Train RMSE")
    ax.plot(train_sizes, test_rmse,  "o-", color="#3B8BD4", label="Test RMSE")
    ax.fill_between(train_sizes, train_rmse - train_std, train_rmse + train_std, alpha=0.1, color="#1D9E75")
    ax.fill_between(train_sizes, test_rmse - test_std,   test_rmse + test_std,   alpha=0.1, color="#3B8BD4")
    ax.axhline(CLINICAL_RMSE_THRESHOLD, color="#E24B4A", linestyle="--", linewidth=1, label=f"Seuil clinique {CLINICAL_RMSE_THRESHOLD} mg/dL")
    ax.set_xlabel("Taille du jeu d'entraînement (fenêtres repas)", fontsize=10)
    ax.set_ylabel("RMSE (mg/dL)", fontsize=10)
    ax.set_title("Courbe d'apprentissage — Random Forest (t+60 min)", fontsize=10)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" ->  Courbe d'apprentissage : {output_path}")


def run_gridsearch_t60(X_v: pd.DataFrame, y_v: pd.Series, g_v: pd.Series):
    """
    GridSearch avec nested CV sur t+60 min uniquement.

    POURQUOI NESTED CV ?
    Sans nested CV, le GridSearch sélectionne les meilleurs hyperparamètres
    sur les mêmes folds utilisés pour évaluer la performance → biais optimiste.
    Avec nested CV :
    - Boucle EXTERNE (GroupKFold k=5) : évalue la performance réelle du
      meilleur modèle sélectionné par la boucle interne → pas de leakage.
    - Boucle INTERNE (GridSearchCV k=3) : sélectionne les hyperparamètres
      optimaux sur le train de chaque fold externe uniquement.
    """
    print("\n  GridSearch RF — t+60 min (nested CV)")

    param_grid = {
        "model__max_depth":    [3, 5, 8, 10],
        "model__n_estimators": [100, 200, 300],
    }

    base_pipeline = Pipeline(
        get_preprocessing() + [("model", RandomForestRegressor(
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ))]
    )

    # Boucle interne : GridSearch sur k=3 folds
    inner_cv = GroupKFold(n_splits=3)
    grid_search = GridSearchCV(
        base_pipeline,
        param_grid,
        cv=inner_cv,
        scoring="neg_root_mean_squared_error",
        refit=True,
        n_jobs=-1,
    )

    # Boucle externe : évaluation réelle sur k=5 folds
    outer_cv = GroupKFold(n_splits=N_FOLDS)
    outer_rmses = []
    best_params_per_fold = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_v, y_v, groups=g_v)):
        X_train, X_test = X_v.iloc[train_idx], X_v.iloc[test_idx]
        y_train, y_test = y_v.iloc[train_idx], y_v.iloc[test_idx]
        g_train = g_v.iloc[train_idx]

        grid_search.fit(X_train, y_train, groups=g_train)
        y_pred = grid_search.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        outer_rmses.append(rmse)
        best_params_per_fold.append(grid_search.best_params_)

        print(f"    Fold {fold_idx+1} | RMSE={rmse:.2f} | "
              f"best_params={grid_search.best_params_}")

    rmse_m = np.mean(outer_rmses)
    rmse_s = np.std(outer_rmses)
    print(f"\n  → GridSearch RF t+60 | RMSE={rmse_m:.2f}±{rmse_s:.2f} mg/dL")

    # Paramètres les plus fréquents sur les 5 folds
    params_df = pd.DataFrame(best_params_per_fold)
    print("\n  Hyperparamètres sélectionnés par fold :")
    print(params_df.to_string(index=False))

    most_common = {
        col: params_df[col].mode()[0] for col in params_df.columns
    }
    print(f"\n  Paramètres les plus fréquents : {most_common}")
    print(f"  → Utiliser ces valeurs pour mettre à jour RF_PARAMS si amélioration confirmée")

    return rmse_m, rmse_s, most_common, params_df



#Pipeline d'entraînement et évaluation
def run():
    from sklearn.model_selection import GroupKFold
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"  TÂCHE 1 — Régression | Modèle : {MODEL_NAME}")
 
    df = load_dataset(DATASET_PATH)
    X=build_X(df)
    groups= df["patient_id"]
 
    print(f"\n  Dataset : {len(df)} fenêtres | {df['patient_id'].nunique()} patients")
    print(f"Features : {X.shape[1]} colonnes (Config A)")
    print(f"Paramètres RF : {RF_PARAMS}")
 
    all_results = []
 
    for horizon_key, target_col in REGRESSION_TARGETS.items():
        y = df[target_col].copy()
        valid= y.notna()
        X_v= X[valid].reset_index(drop=True)
        y_v= y[valid].reset_index(drop=True)
        g_v =groups[valid].reset_index(drop=True)
        print(f"\n  Horizon {horizon_key} ({target_col}) — {valid.sum()} fenêtres valides")
 
        pipeline = Pipeline(
            get_preprocessing() + [("model", RandomForestRegressor(**RF_PARAMS))]
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
 
            print(f" Fold {fold_idx+1} | RMSE={fold_result['rmse']:.2f} | "
                  f"MAE={fold_result['mae']:.2f} | R²={fold_result['r2']:.3f}")
        rmse_m, rmse_s = np.mean(rmses), np.std(rmses)
        mae_m,  mae_s  = np.mean(maes),  np.std(maes)
        r2_m,   r2_s   = np.mean(r2s),   np.std(r2s)
        print(f"\n  → Moyenne | RMSE={rmse_m:.2f}±{rmse_s:.2f} | "
              f"MAE={mae_m:.2f}±{mae_s:.2f} | R²={r2_m:.3f}±{r2_s:.3f}")
 
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
        # Courbe d'apprentissage uniquement pour t+60 (coûteux en calcul)
        if horizon_key == "t60":
            plot_learning_curve(
                pipeline, X_v, y_v, g_v,
                output_path=os.path.join(OUTPUT_DIR, "learning_curve_t60.png"),
            )
            # --- GRIDSEARCH nested CV ---
            gs_rmse_m, gs_rmse_s, best_params, params_df = run_gridsearch_t60(X_v, y_v, g_v)
            params_df.to_csv(
                os.path.join(OUTPUT_DIR, "gridsearch_best_params_t60.csv"), index=False
            )
            print(f"\n  Comparaison RF fixe vs GridSearch à t+60 :")
            print(f"  RF fixe    : RMSE={rmse_m:.2f}±{rmse_s:.2f}")
            print(f"  RF GridSearch : RMSE={gs_rmse_m:.2f}±{gs_rmse_s:.2f}")
 
        all_results.append({
            "model": MODEL_NAME,
            "task":  "regression",
            "horizon": f"t+{horizon_key[1:]} min",
            "rmse_mean":  round(rmse_m, 2),
            "rmse_std":round(rmse_s, 2),
            "mae_mean": round(mae_m, 2),
            "mae_std":round(mae_s, 2),
            "r2_mean":round(r2_m, 3),
            "r2_std": round(r2_s, 3),
            "n_features": X_v.shape[1],
            "n_estimators": RF_PARAMS["n_estimators"],
            "max_depth":  RF_PARAMS["max_depth"],
            "clinical_ok": rmse_m <= CLINICAL_RMSE_THRESHOLD,
        })
 
    save_results(all_results, os.path.join(OUTPUT_DIR, "results_random_forest_regression.csv"))
 
 
if __name__ == "__main__":
    run()







