"""
task1_linear_models.py

Régression continue : comparaison des modèles linéaires.

Ce script répond à trois questions posées par Mr Redjdal :
1- Quelle est la vraie baseline la plus simple ?
-->  Régression linéaire ordinaire (OLS), sans aucune pénalisation.
La Ridge n'est pas une baseline : c'est déjà un modèle régularisé.

2- Le Lasso fait-il mieux ou moins bien que la Ridge ?
--> Le Lasso (pénalisation L1) force certains coefficients à ZÉRO, ce qui 
revient à faire une sélection automatique de variables.
C'est différent de la Ridge (L2) qui réduit tous les coefficients
sans en annuler aucun.

3- Est ce qu'une pipeline Lasso : Random Forest améliore les résultats ?
--> On utilise le Lasso pour identifier les features pertinentes,
puis on entraîne un Random Forest uniquement sur ces features.
L'idée : donner au RF un espace de features "nettoyé" pour réduire le bruit
 et améliorer la généralisation.


Ce code permet de sortir : 
- Métriques comparatives des 3 modèles linéaires (OLS, Ridge, Lasso)
- Graphique de comparaison des coefficients côte à côte
- Liste des variables sélectionnées par le Lasso (coeff ≠ 0)
Pipeline Lasso → Random Forest avec ses métriques
- CSV de résultats compatible avec compare_task1_regression.py

La validation se fait toujours en GroupKFold strictement par patient, pour éviter tout data leakage.
Avec la configuration A : agrégats CGM pré-repas uniquement, pour rester cohérent avec les autres modèles.

Auteurs : Palliere Raphael & Bouny Mathieu — E4 Bio
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
from scipy import stats #biblio calcul scientifique basé sur NumPy.
from sklearn.linear_model import LinearRegression, Ridge, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
 
from config import (
    DATASET_PATH, RESULTS_DIR, RANDOM_STATE,
    CLINICAL_RMSE_THRESHOLD, REGRESSION_TARGETS,
    load_dataset, build_X, get_preprocessing, get_cv_splits, save_results,
)
 
warnings.filterwarnings("ignore")

 
OUTPUT_DIR = os.path.join(RESULTS_DIR, "task1_linear_models")


# Explication des Hyperparamètres des modèles linéaires
# Ridge alpha=1.0 : valeur par défaut raisonnable. Une valeur plus élevée renforce la régularisation (coefficients plus petits), une valeur plus faible se rapproche de l'OLS. On garde 1.0 pour la comparabilité.
#
# LassoCV : on utilise LassoCV plutôt que Lasso(alpha=X) car LassoCV sélectionne AUTOMATIQUEMENT le meilleur alpha par validation croisée INTERNE au jeu d'entraînement. On ne touche pas au jeu de test.
# cv=5 : 5 folds internes pour choisir alpha. max_iter=5000 : nécessaire pour la convergence sur des données normalisées.
#
# Random Forest n_estimators=200, max_depth=8 : mêmes paramètres que dans
# task1_random_forest.py pour garantir la comparabilité.

RIDGE_PARAMS = {"alpha": 1.0, "random_state": RANDOM_STATE}
LASSO_CV_PARAMS = {"cv": 5, "max_iter": 5000, "random_state": RANDOM_STATE}
RF_PARAMS = {
    "n_estimators": 200, "max_depth": 8,
    "min_samples_leaf": 5, "max_features": "sqrt",
    "random_state": RANDOM_STATE, "n_jobs": -1,
}


#Evaluation par FOld : 
def evaluate_fold(pipeline, X_train, y_train, X_test, y_test) -> dict:
    """
    Entraîne le pipeline sur le fold d'entraînement et évalue sur le test.
    Le fit est fait Ici pour que l'imputation et la normalisation ne voient
    jamais les données de test — condition sine qua non pour éviter le leakage.
    """
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    return {
        "rmse":float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2":float(r2_score(y_test, y_pred)),
        "y_test": y_test.values,
        "y_pred": y_pred,
    }



def run_model(model_name: str, pipeline: Pipeline, X_v: pd.DataFrame, y_v: pd.Series, g_v: pd.Series,horizon_key: str) -> dict:
    """
    Lance la validation croisée GroupKFold pour un modèle donné.
    Retourne un dictionnaire de résultats agrégés sur les 5 folds.
    """
    splits = get_cv_splits(X_v, y_v, g_v)
    rmses, maes, r2s = [], [], []
    last_y_test, last_y_pred = None, None
 
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        res = evaluate_fold(
            pipeline,
            X_v.iloc[train_idx], y_v.iloc[train_idx],
            X_v.iloc[test_idx],  y_v.iloc[test_idx],
        )
        rmses.append(res["rmse"])
        maes.append(res["mae"])
        r2s.append(res["r2"])
        last_y_test = res["y_test"]
        last_y_pred = res["y_pred"]
        print(f" Fold {fold_idx+1} | RMSE={res['rmse']:.2f} | "
              f"MAE={res['mae']:.2f} | R²={res['r2']:.3f}")
 
    rmse_m, rmse_s = np.mean(rmses), np.std(rmses)
    mae_m,  mae_s = np.mean(maes),  np.std(maes)
    r2_m,   r2_s = np.mean(r2s),   np.std(r2s)
 
    print(f"\n  -< {model_name} | RMSE={rmse_m:.2f}±{rmse_s:.2f} | "
          f"MAE={mae_m:.2f}±{mae_s:.2f} | R²={r2_m:.3f}±{r2_s:.3f}")
 
    return {
        "model":model_name,
        "task": "regression",
        "horizon":  f"t+{horizon_key[1:]} min",
        "rmse_mean":  round(rmse_m, 2),
        "rmse_std": round(rmse_s, 2),
        "mae_mean": round(mae_m, 2),
        "mae_std":  round(mae_s, 2),
        "r2_mean":  round(r2_m, 3),
        "r2_std": round(r2_s, 3),
        "n_features": X_v.shape[1],
        "clinical_ok": rmse_m <= CLINICAL_RMSE_THRESHOLD,
        "y_test":  last_y_test,
        "y_pred":  last_y_pred,
        "pipeline": pipeline,
    }
 
 

# On extrait les coefficients 
def get_coefficients(pipeline: Pipeline, feature_names: list, model_key: str) -> pd.DataFrame:
    """
    Extrait les coefficients du modèle linéaire après entraînement.
 
    POURQUOI COMPARER LES COEFFICIENTS ?
    En régression linéaire, chaque coefficient représente l'effet marginal
    d'une feature sur la cible (après normalisation, ils sont comparables).
    - OLS  : coefficients "purs", sans contrainte. Peuvent être très grands
        si les features sont colinéaires (instabilité numérique).
    - Ridge: coefficients réduits uniformément vers 0 par la pénalisation L2.
        Aucun n'est exactement nul — Ridge ne fait pas de sélection.
    - Lasso: coefficients forcés à EXACTEMENT 0 pour les features non utiles.
        C'est la signature de la sélection automatique de variables.
 
    Comparer les trois côte à côte permet de voir quelles features sont
    stables (présentes dans les 3), lesquelles disparaissent avec Lasso,
    et lesquelles sont amplifiées ou réduites par la régularisation.
    """
    model = pipeline.named_steps["model"]
    coefs = model.coef_
 
    return pd.DataFrame({
        "feature":          feature_names,
        f"coef_{model_key}": coefs,
    })
 
 
def get_lasso_selected_features(pipeline: Pipeline, feature_names: list) -> list:
    """
    Retourne la liste des features dont le coefficient Lasso est NON NUL.
 
    POURQUOI ?
    Le Lasso force exactement à 0 les features qu'il juge non pertinentes.
    Les features restantes (coeff ≠ 0) sont celles que le modèle considère
    comme vraiment utiles pour la prédiction.
    C'est une sélection de variables automatique et statistiquement fondée,
    qui complète et valide notre sélection manuelle basée sur l'importance Gini.
    """
    model = pipeline.named_steps["model"]
    coefs = model.coef_
 
    selected = [f for f, c in zip(feature_names, coefs) if abs(c) > 1e-6]
    eliminated = [f for f, c in zip(feature_names, coefs) if abs(c) <= 1e-6]
 
    print(f"\n  Features sélectionnés par Lasso ({len(selected)}/{len(feature_names)}) :")
    for f in selected:
        c = coefs[feature_names.index(f)]
        print(f" {f:<40} coeff = {c:+.4f}")
 
    print(f"\n  Features ÉLIMINÉES par Lasso ({len(eliminated)}) :")
    for f in eliminated:
        print(f"  {f}")
    return selected

def plot_residuals_diagnostics(y_test: np.ndarray, y_pred: np.ndarray, horizon_label: str, output_path: str):
    """
    Diagnostics des résidus OLS sur le dernier fold :
    - Résidus vs valeurs prédites (homoscédasticité)
    - QQ-plot (normalité)
    - Test de Shapiro-Wilk (normalité formelle)

    POURQUOI CES TESTS ?
    La régression linéaire suppose :
    1. Homoscédasticité : la variance des résidus est constante quelle que
       soit la valeur prédite. Un entonnoir dans le plot résidus/prédits
       révèle une hétéroscédasticité → les IC et p-values sont invalides.
    2. Normalité des résidus : nécessaire pour que les tests t sur les
       coefficients soient valides. Shapiro-Wilk teste H0 = normalité.
       Si p < 0.05, on rejette la normalité.
    """
    residuals = y_test - y_pred

    # Test de Shapiro-Wilk (limité à 5000 points max par contrainte scipy)
    n = len(residuals)
    sample = residuals if n <= 5000 else np.random.choice(residuals, 5000, replace=False)
    stat, p_value = stats.shapiro(sample)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1 : Résidus vs Prédits (homoscédasticité) ---
    ax1 = axes[0]
    ax1.scatter(y_pred, residuals, alpha=0.45, color="#3B8BD4", s=25)
    ax1.axhline(0, color="black", linewidth=1, linestyle="--")
    ax1.axhline( 15, color="#EF9F27", linewidth=0.8, linestyle=":", alpha=0.7)
    ax1.axhline(-15, color="#EF9F27", linewidth=0.8, linestyle=":", alpha=0.7, label="±15 mg/dL (seuil ISO)")
    ax1.set_xlabel("Valeurs prédites (mg/dL)", fontsize=10)
    ax1.set_ylabel("Résidus (réel − prédit, mg/dL)", fontsize=10)
    ax1.set_title(f"Résidus vs Prédits — OLS {horizon_label}\n" f"(homoscédasticité : pas d'entonnoir attendu)", fontsize=10)
    ax1.legend(fontsize=8)

    # Plot 2 : QQ-plot (normalité) 
    ax2 = axes[1]
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
    ax2.plot(osm, osr, "o", alpha=0.45, color="#3B8BD4", markersize=3)
    ax2.plot(osm, slope * np.array(osm) + intercept, color="#E24B4A", linewidth=1.5, label="Droite théorique normale")
    ax2.set_xlabel("Quantiles théoriques", fontsize=10)
    ax2.set_ylabel("Quantiles observés", fontsize=10)
    shapiro_label = (f"Shapiro-Wilk : W={stat:.4f}, p={p_value:.4f}\n" f"{'✅ Normalité non rejetée (p≥0.05)' if p_value >= 0.05 else '⚠️  Normalité rejetée (p<0.05)'}")
    ax2.set_title(f"QQ-plot des résidus — OLS {horizon_label}\n{shapiro_label}", fontsize=10)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" -> Diagnostics résidus : {output_path}")
    print(f" Shapiro-Wilk W={stat:.4f} | p={p_value:.4f} | n={n}")
    if p_value < 0.05:
        print(f"⚠️  Normalité des résidus rejetée : Il faut donc interpréter les IC avec prudence")
    else:
        print(f"✅ Normalité des résidus non rejetée")


# Extraction des coefficients pour les 3 modèles linéaires et comparaison graphique
def get_coefficients(pipeline: Pipeline, feature_names: list, model_key: str) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    coefs = model.coef_
 
    return pd.DataFrame({
        "feature": feature_names,
        f"coef_{model_key}": coefs,
    })
 
 
def get_lasso_selected_features(pipeline: Pipeline, feature_names: list) -> list:
    """
    Retourne la liste des features dont le coefficient Lasso est non nul.
    """
    model = pipeline.named_steps["model"]
    coefs = model.coef_
 
    selected = [f for f, c in zip(feature_names, coefs) if abs(c) > 1e-6]
    eliminated = [f for f, c in zip(feature_names, coefs) if abs(c) <= 1e-6]
 
    print(f"\n  Features SÉLECTIONNÉES par Lasso ({len(selected)}/{len(feature_names)}) :")
    for f in selected:
        c = coefs[feature_names.index(f)]
        print(f"{f:<40} coeff = {c:+.4f}")
 
    print(f"\n  Features ÉLIMINÉES par Lasso ({len(eliminated)}) :")
    for f in eliminated:
        print(f"{f}")
    return selected
 
 


#Visualisation : 
def plot_coefficients_comparison(df_ols: pd.DataFrame, df_ridge: pd.DataFrame, df_lasso: pd.DataFrame, horizon: str, output_path: str):
    """
    Graphique de comparaison des coefficients OLS vs Ridge vs Lasso.
    Il y a :
    - Chaque ligne = une feature
    - Les trois barres = le coefficient dans chaque modèle
    - Si une barre Lasso est absente (= 0) : la feature a été éliminée
    - Si Ridge < OLS : la pénalisation a réduit l'effet de cette feature
    - Un coefficient positif = la feature augmente la glycémie prédite
    - Un coefficient négatif = la feature diminue la glycémie prédite
 
    Attention : les coefficients sont calculés après normalisation des features
    (StandardScaler), donc ils sont comparables entre eux en termes d'importance.
    """
    # Fusionner les 3 DataFrames
    merged = df_ols.merge(df_ridge, on="feature").merge(df_lasso, on="feature")
 
    # Trier par importance absolue OLS
    merged["abs_ols"] = merged["coef_OLS"].abs()
    merged = merged.sort_values("abs_ols", ascending=True).tail(20)
 
    features = merged["feature"].tolist()
    y_pos = np.arange(len(features))
    height = 0.25
 
    fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.4)))
 
    ax.barh(y_pos-height, merged["coef_OLS"],  height, label="OLS",   color="#888780", alpha=0.85)
    ax.barh(y_pos,merged["coef_Ridge"],  height, label="Ridge", color="#3B8BD4", alpha=0.85)
    ax.barh(y_pos + height, merged["coef_Lasso"], height, label="Lasso", color="#1D9E75", alpha=0.85)
 
    ax.axvline(0, color="black", linewidth=0.8, linestyle="-")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=9)
    ax.set_xlabel("Coefficient (features normalisées)", fontsize=10)
    ax.set_title(
        f"Comparaison des coefficients — OLS / Ridge / Lasso\n"
        f"Horizon {horizon} | Features normalisées (comparables entre elles)",
        fontsize=10
    )
    ax.legend(fontsize=9)
 
    # Annoter les features éliminées par le Lasso (coeff = 0)
    for i, row in enumerate(merged.itertuples()):
        if abs(row.coef_Lasso) <= 1e-6:
            ax.text(0.02, y_pos[i] + height, "Lasso=0", fontsize=7, color="#E24B4A", va="center")
 
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" -> Comparaison coefficients : {output_path}")
 
 

def plot_lasso_alpha_path(pipeline: Pipeline, feature_names: list, horizon: str, output_path: str):
    """
    Visualise l'alpha sélectionné par LassoCV et le nombre de features retenues.
    LassoCV teste automatiquement plusieurs valeurs d'alpha et choisit
    celle qui minimise l'erreur de validation croisée interne.
    Un alpha faible = peu de régularisation (beaucoup de features retenues).
    Un alpha élevé = régularisation forte (peu de features retenues).
    Ce graphique montre où se situe l'alpha optimal choisi automatiquement.
    """
    lasso = pipeline.named_steps["model"]
    alpha_opt = lasso.alpha_
    n_nonzero = np.sum(np.abs(lasso.coef_) > 1e-6)
 
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axvline(alpha_opt, color="#E24B4A", linewidth=2, label=f"Alpha optimal = {alpha_opt:.4f}")
    ax.text(alpha_opt * 1.1, 0.5, f"{n_nonzero} features\nretenues", transform=ax.get_xaxis_transform(), fontsize=9, color="#E24B4A")
    ax.set_xlabel("Alpha (force de régularisation)", fontsize=10)
    ax.set_title(f"Lasso — Alpha sélectionné automatiquement\nHorizon {horizon}", fontsize=10)
    ax.legend(fontsize=9)
    ax.set_xlim(0, alpha_opt * 5 + 0.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" -> Alpha Lasso : {output_path}")
 


def plot_scatter(y_test, y_pred, title: str, color: str, output_path: str):
    """Scatter prédit vs réel standardisé."""
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
 
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, alpha=0.55, color=color, s=35)
    lims = [min(y_test.min(), y_pred.min()) - 5, max(y_test.max(), y_pred.max()) + 5]
    ax.plot(lims, lims, "k--", linewidth=1, label="Prédiction parfaite")
    ax.fill_between(lims, [l - 15 for l in lims], [l + 15 for l in lims], alpha=0.1, color=color, label="±15 mg/dL")
    for seuil, label, c in [(70, "Hypo<70", "#E24B4A"), (140, "Hyper>140", "#EF9F27")]:
        ax.axhline(seuil, color=c, linestyle=":", linewidth=0.8, alpha=0.7)
        ax.text(lims[0] + 1, seuil + 1, label, fontsize=8, color=c)
    ax.set_xlabel("Glycémie réelle (mg/dL)", fontsize=10)
    ax.set_ylabel("Glycémie prédite (mg/dL)", fontsize=10)
    ax.set_title(f"{title}\nRMSE={rmse:.1f} | MAE={mae:.1f} mg/dL", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_xlim(lims); ax.set_ylim(lims)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
 

def plot_lasso_rf_feature_importance(pipeline: Pipeline, feature_names: list, horizon: str, output_path: str):
    """
    Importance des features du Random Forest entraîné sur les features Lasso.
    Après sélection par le Lasso, le Random Forest n'a accès qu'aux features
    jugées pertinentes. Ce graphique montre si le RF s'appuie sur les mêmes
    features que le Lasso ou s'il en utilise certaines différemment.
    C'est la validation de la cohérence inter-modèles demandée par l'encadrant.
    """
    rf = pipeline.named_steps["rf"]
    importances = rf.feature_importances_
    std_imp = np.std(
        [t.feature_importances_ for t in rf.estimators_], axis=0
    )
    idx = np.argsort(importances)[::-1]
 
    fig, ax = plt.subplots(figsize=(8, max(5, len(feature_names) * 0.35)))
    ax.barh(
        [feature_names[i] for i in idx][::-1],
        importances[idx][::-1],
        xerr=std_imp[idx][::-1],
        color="#1D9E75", alpha=0.85, capsize=3,
    )
    ax.set_xlabel("Importance (réduction d'impureté)", fontsize=10)
    ax.set_title(
        f"Pipeline Lasso -> RF — Importance des features\n"
        f"Horizon {horizon} | Uniquement les features sélectionnées par Lasso",
        fontsize=10
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" -> Importance Lasso -> RF : {output_path}")
 

#Pipeline LAsso --> Random FOrest
class LassoFeatureSelector:
    """
    Transformateur sklearn-compatible qui :
      1. Entraîne un LassoCV sur le jeu d'entraînement
      2. Identifie les features avec coefficient ≠ 0
      3. Ne retient que ces features pour la suite du pipeline
 
    POURQUOI faire une classe dédiée ?
    Pour intégrer la sélection Lasso dans un Pipeline sklearn, il faut un objet
    qui implémente fit() et transform(). Ainsi, la sélection est faite
    UNIQUEMENT sur le jeu d'entraînement de chaque fold — jamais sur le test.
    C'est ce qui garantit l'absence de data leakage dans la sélection elle-même.
 
    Sans cette précaution, si on sélectionnait les features sur l'ensemble du
    dataset puis qu'on entraînait le RF, on aurait une fuite d'information :
    le choix des features serait influencé par les données de test.
    """
 
    def __init__(self, lasso_params: dict):
        self.lasso_params = lasso_params
        self.selected_indices_ = None
        self.selected_features_ = None
        self.lasso_ = None
 
    def fit(self, X, y):
        """
        Entraîne le Lasso sur X_train, identifie les features non nulles.
        X peut être un array numpy (après StandardScaler).
        """
        lasso = LassoCV(**self.lasso_params)
        lasso.fit(X, y)
        self.lasso_ = lasso
        # Indices des features avec coefficient non nul
        self.selected_indices_ = np.where(np.abs(lasso.coef_) > 1e-6)[0]
        return self
 
    def transform(self, X):
        """Retourne uniquement les colonnes sélectionnées."""
        if self.selected_indices_ is None:
            raise RuntimeError("Appeler fit() avant transform()")
        return X[:, self.selected_indices_]
 
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
 
    def get_selected_feature_names(self, feature_names: list) -> list:
        """Retourne les noms des features sélectionnées."""
        return [feature_names[i] for i in self.selected_indices_]
 

def build_lasso_rf_pipeline() -> Pipeline:
    """
    Construit le pipeline Lasso → Random Forest.
    1. Imputation  : remplace les NaN par la médiane (calculée sur train)
    2. Normalisation: StandardScaler (nécessaire pour que le Lasso fonctionne)
    3. Sélecteur : LassoCV — sélectionne les features pertinentes
    4. RF  : Random Forest entraîné uniquement sur les features retenues

    """
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
 
    selector = LassoFeatureSelector(lasso_params=LASSO_CV_PARAMS)
 
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("selector", selector),
        ("rf",  RandomForestRegressor(**RF_PARAMS)),
    ])
    return pipeline
 

#Pipeline principal : 
def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(" 1 — Comparaison modèles linéaires + Pipeline Lasso -> RF")
 
    df = load_dataset(DATASET_PATH)
    X = build_X(df)
    groups = df["patient_id"]
    feature_names = list(X.columns)
 
    print(f"\n Dataset : {len(df)} fenêtres | {df['patient_id'].nunique()} patients")
    print(f" Features : {X.shape[1]} colonnes (Config A)")
    all_results = []




# Boucle sur les horizons de prédiction
    for horizon_key, target_col in REGRESSION_TARGETS.items():
            y  = df[target_col].copy()
            valid = y.notna()
            X_v = X[valid].reset_index(drop=True)
            y_v = y[valid].reset_index(drop=True)
            g_v = groups[valid].reset_index(drop=True)
            horizon_label = f"t+{horizon_key[1:]} min"
 
            print(f"  HORIZON {horizon_key} ({target_col}) — {valid.sum()} fenêtres")
 


# 1. OLS (baseline simple)
            print(f"\n [1/4] OLS — Régression linéaire ordinaire (baseline)")
            print(f" Aucune pénalisation : c'est le modèle le plus simple possible.")
            print(f" Sert de référence absolue pour juger l'apport de la régularisation.")
 
            ols_pipeline = Pipeline(
                get_preprocessing() + [("model", LinearRegression())]
            )
            res_ols = run_model("OLS", ols_pipeline, X_v, y_v, g_v, horizon_key)
            all_results.append({k: v for k, v in res_ols.items()
                if k not in ["y_test", "y_pred", "pipeline"]})
 
            plot_scatter(
                res_ols["y_test"], res_ols["y_pred"],
                f"OLS (baseline) — {horizon_label}", "#888780",
                os.path.join(OUTPUT_DIR, f"scatter_OLS_{horizon_key}.png"),
            )


            # Diagnostics résidus OLS (Shapiro-Wilk + QQ-plot)
            plot_residuals_diagnostics(
                res_ols["y_test"], res_ols["y_pred"], horizon_label, output_path=os.path.join(OUTPUT_DIR, f"residuals_OLS_{horizon_key}.png"),
)
 

# 2. Ridge
            print(f"\n  [2/4] Ridge — Régularisation L2 (alpha={RIDGE_PARAMS['alpha']})")
            print(f" Réduit tous les coefficients vers 0 sans en annuler aucun.")
            print(f" Utile quand les features sont colinéaires (ex: cgm_pre_mean et cgm_at_meal).")
 
            ridge_pipeline = Pipeline(
            get_preprocessing() + [("model", Ridge(**RIDGE_PARAMS))]
            )
            res_ridge = run_model("Ridge", ridge_pipeline, X_v, y_v, g_v, horizon_key)
            all_results.append({k: v for k, v in res_ridge.items()
                if k not in ["y_test", "y_pred", "pipeline"]})
 
            plot_scatter(
                res_ridge["y_test"], res_ridge["y_pred"],
                f"Ridge — {horizon_label}", "#3B8BD4",
                os.path.join(OUTPUT_DIR, f"scatter_Ridge_{horizon_key}.png"),
            )

# 3. Lasso
            print(f"\n [3/4] Lasso — Régularisation L1 (alpha choisi automatiquement par LassoCV)")
            print(f" Force certains coefficients EXACTEMENT à 0 = sélection de variables.")
            print(f" LassoCV choisit le meilleur alpha par CV interne sur le jeu d'entraînement.")
            lasso_pipeline = Pipeline(
                get_preprocessing() + [("model", LassoCV(**LASSO_CV_PARAMS))]
            )
            res_lasso = run_model("Lasso", lasso_pipeline, X_v, y_v, g_v, horizon_key)
            all_results.append({k: v for k, v in res_lasso.items()
            if k not in ["y_test", "y_pred", "pipeline"]})
            plot_scatter(
                res_lasso["y_test"], res_lasso["y_pred"],
                f"Lasso — {horizon_label}", "#EF9F27",
                os.path.join(OUTPUT_DIR, f"scatter_Lasso_{horizon_key}.png"),
            )
 
        # Alpha retenu par LassoCV (sur le dernier fold — indicatif)
            lasso_model = res_lasso["pipeline"].named_steps["model"]
            print(f"\n Alpha Lasso retenu (dernier fold) : {lasso_model.alpha_:.4f}")
            plot_lasso_alpha_path(
                res_lasso["pipeline"], feature_names, horizon_label,
                os.path.join(OUTPUT_DIR, f"lasso_alpha_{horizon_key}.png"),
            )
 
            # Features sélectionnées par le Lasso (sur le dernier fold — indicatif)
            lasso_selected = get_lasso_selected_features(
                res_lasso["pipeline"], feature_names
            )
 
        # Sauvegarder la liste des features sélectionnées
            pd.DataFrame({
                "feature": feature_names,
                "coef_lasso": lasso_model.coef_,
                "selected": [abs(c) > 1e-6 for c in lasso_model.coef_],
            }).sort_values("selected", ascending=False).to_csv(
                os.path.join(OUTPUT_DIR, f"lasso_features_{horizon_key}.csv"),
                index=False,
            )


# Graphique de comparaison des coefficients OLS vs Ridge vs Lasso
            print(f"\n  Génération du graphique comparatif des coefficients...")
        # Ré-entraîner sur tout le dataset pour avoir les coefficients
        # représentatifs (pas seulement le dernier fold)
            for pipe, name in [(res_ols["pipeline"], "OLS"), (res_ridge["pipeline"], "Ridge"), (res_lasso["pipeline"], "Lasso")]:
            # Imputer + scaler + fit sur toutes les données
                pipe.fit(X_v, y_v)
 
            df_coef_ols   = get_coefficients(res_ols["pipeline"],   feature_names, "OLS")
            df_coef_ridge = get_coefficients(res_ridge["pipeline"], feature_names, "Ridge")
            df_coef_lasso = get_coefficients(res_lasso["pipeline"], feature_names, "Lasso")
 
            plot_coefficients_comparison(
                df_coef_ols, df_coef_ridge, df_coef_lasso, horizon_label,
                output_path=os.path.join(OUTPUT_DIR, f"coefficients_comparison_{horizon_key}.png"),
            )
 
            # Sauvegarder le tableau comparatif des coefficients
            coef_table = df_coef_ols.merge(df_coef_ridge, on="feature").merge(df_coef_lasso, on="feature")
            coef_table["lasso_selected"] = coef_table["coef_Lasso"].abs() > 1e-6
            coef_table.to_csv(
                os.path.join(OUTPUT_DIR, f"coef_table_{horizon_key}.csv"), index=False
            )


# 4. Pipeline Lasso → Random Forest
            print(f"\n  [4/4] Pipeline Lasso → Random Forest")
            print(f" Étape 1 (dans chaque fold) : Lasso sélectionne les features pertinentes")
            print(f"Étape 2 (dans chaque fold) : Random Forest entraîné uniquement dessus")
            print(f" Objectif : donner au RF un espace de features 'nettoyé' par le Lasso.")
 
            lasso_rf_pipeline = build_lasso_rf_pipeline()
            res_lasso_rf = run_model(
                "Lasso -> RF", lasso_rf_pipeline, X_v, y_v, g_v, horizon_key
            )
            all_results.append({k: v for k, v in res_lasso_rf.items()
            if k not in ["y_test", "y_pred", "pipeline"]})
 
            plot_scatter(
                res_lasso_rf["y_test"], res_lasso_rf["y_pred"],
                f"Pipeline Lasso→RF — {horizon_label}", "#BA7517",
                os.path.join(OUTPUT_DIR, f"scatter_LassoRF_{horizon_key}.png"),
            )
 
        # Importance des features dans le RF (sur le dernier fold)
        # Récupérer les noms des features sélectionnées dans le dernier fold
            selector = res_lasso_rf["pipeline"].named_steps["selector"]
            selected_names = selector.get_selected_feature_names(feature_names)
            n_selected = len(selected_names)
            print(f"\n  Features sélectionnées par Lasso dans ce pipeline : {n_selected}/{len(feature_names)}")
            for f in selected_names:
                print(f"    · {f}")
 
            if n_selected > 0:
                plot_lasso_rf_feature_importance(
                    res_lasso_rf["pipeline"], selected_names, horizon_label,
                    output_path=os.path.join(OUTPUT_DIR, f"importance_LassoRF_{horizon_key}.png"),
                )
 



            save_results(
        all_results,
        os.path.join(OUTPUT_DIR, "results_linear_models.csv"),
    )
            
# Résultats finaux : CSV de tous les modèles pour comparaison
    print(" RÉSUMÉ COMPARATIF — Modèles linéaires + Lasso -> RF")
 
    df_res = pd.DataFrame(all_results)
    for h in ["t+30 min", "t+60 min", "t+90 min"]:
        h_df = df_res[df_res["horizon"] == h]
        if h_df.empty:
            continue
        print(f"\n  Horizon {h} :")
        print(f" {'Modèle':<20} {'RMSE':>14} {'MAE':>14} {'R²':>10}")
        for _, row in h_df.iterrows():
            ok = "✅" if row["clinical_ok"] else "⚠️ "
            print(f" {row['model']:<20} "
                f"{row['rmse_mean']:>6.2f}±{row['rmse_std']:<5.2f} "
                f"{row['mae_mean']:>6.2f}±{row['mae_std']:<5.2f} "
                f"{row['r2_mean']:>8.3f} {ok}")
 
 
if __name__ == "__main__":
    run()
