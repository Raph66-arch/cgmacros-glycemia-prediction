"""
Comparaison des résultats de la Tâche 1 — Régression continue.
  - Régression linéaire Ridge (baseline)
  - Arbre de décision
  - Random Forest

 Le code produit :
  - Un tableau de synthèse complet (CSV)
  - Des graphiques de comparaison (RMSE, MAE, R²) par horizon

  
AUteurs : Palliere Raphael & Bouny Mathieu — E4 Bio

"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
from config import RESULTS_DIR, CLINICAL_RMSE_THRESHOLD
 
warnings.filterwarnings("ignore")
 
OUTPUT_DIR = os.path.join(RESULTS_DIR, "comparison_task1_regression")
 
# Fichiers CSV de résultats à agréger
RESULT_FILES = {
    "Ridge (baseline)": os.path.join(RESULTS_DIR, "baseline_linear",    "baseline_results.csv"),
    "OLS":              os.path.join(RESULTS_DIR, "task1_linear_models", "results_linear_models.csv"),
    "Ridge":            os.path.join(RESULTS_DIR, "task1_linear_models", "results_linear_models.csv"),
    "Lasso":            os.path.join(RESULTS_DIR, "task1_linear_models", "results_linear_models.csv"),
    "Lasso -> RF":      os.path.join(RESULTS_DIR, "task1_linear_models", "results_linear_models.csv"),
    "DecisionTree":     os.path.join(RESULTS_DIR, "task1_decision_tree", "results_decision_tree_regression.csv"),
    "RandomForest":     os.path.join(RESULTS_DIR, "task1_random_forest", "results_random_forest_regression.csv"),
}

COLORS = {
    "Ridge (baseline)": "#888780",
    "OLS":              "#C0C0C0",
    "Ridge":            "#3B8BD4",
    "Lasso":            "#EF9F27",
    "Lasso -> RF":      "#BA7517",
    "DecisionTree":     "#9B59B6",
    "RandomForest":     "#1D9E75",
}
#task1_linear_models.py stocke les 4 modèles dans un seul fichier results_linear_models.csv,
# avec une colonne model qui vaut "OLS", "Ridge", "Lasso" ou "Lasso -> RF". La fonction load_all_results dans compare_task1_regression.py
# filtre déjà par model_label quand elle construit les graphiques — donc chaque modèle sera bien isolé à l'affichage.
 
HORIZONS = ["t+30 min", "t+60 min", "t+90 min"]
COLORS = {"Ridge (baseline)": "#888780", "DecisionTree": "#3B8BD4", "RandomForest": "#1D9E75"}



#Fonction d'agrégation des résultats CSV de chaque modèle
def load_all_results() -> pd.DataFrame:
    frames = []
    for model_name, path in RESULT_FILES.items():
        df = pd.read_csv(path)
        # Filtrer uniquement la ligne correspondant à ce modèle
        if "model" in df.columns:
            df = df[df["model"] == model_name].copy()
        df["model_label"] = model_name
        frames.append(df)
 
    combined = pd.concat(frames, ignore_index=True)
 
    # Normaliser le nom de la colonne horizon
    if "horizon" not in combined.columns and "Horizon" in combined.columns:
        combined = combined.rename(columns={"Horizon": "horizon"})
    return combined
 

# Visualisation 
def plot_metric_comparison(df: pd.DataFrame, metric: str, metric_label: str, std_col: str, output_path: str):
    """
    Barplot groupé : chaque groupe = un horizon, chaque barre = un modèle.
    Inclut les barres d'erreur (écart-type inter-folds).
    """
    models= df["model_label"].unique().tolist()
    horizons = [h for h in HORIZONS if h in df["horizon"].values]
    x= np.arange(len(horizons))
    width = 0.25
    n = len(models)
    offsets= np.linspace(-(n - 1) * width / 2, (n - 1) * width / 2, n)
 
    fig, ax= plt.subplots(figsize=(9, 5))
 
    for i, model in enumerate(models):
        model_df = df[df["model_label"] == model]
        values= []
        stds  =[]
        for h in horizons:
            row = model_df[model_df["horizon"] == h]
            values.append(float(row[metric].values[0]) if len(row) > 0 else 0)
            stds.append(float(row[std_col].values[0]) if len(row) > 0 else 0)
 
        ax.bar(
            x + offsets[i], values, width,
            yerr=stds, capsize=4,
            label=model,
            color=COLORS.get(model, f"C{i}"),
            alpha=0.85,
        )
 
    # Seuil clinique uniquement pour le RMSE
    if metric == "rmse_mean":
        ax.axhline(CLINICAL_RMSE_THRESHOLD, color="#E24B4A", linestyle="--", linewidth=1.2, label=f"Seuil clinique ISO 15197 ({CLINICAL_RMSE_THRESHOLD} mg/dL)")
 
    ax.set_xticks(x)
    ax.set_xticklabels(horizons)
    ax.set_ylabel(metric_label, fontsize=11)
    ax.set_title(f"Tâche 1 — Régression continue\n{metric_label} par modèle et horizon", fontsize=11)
    ax.legend(fontsize=9)
 
    #Annoter la meilleure valeur par horizon
    for j, h in enumerate(horizons):
        h_df = df[df["horizon"] == h]
        if metric == "r2_mean":
            best_idx = h_df[metric].idxmax()
        else:
            best_idx = h_df[metric].idxmin()
        best_val = h_df.loc[best_idx, metric]
        ax.text(j, best_val + 0.3, f"★", ha="center", fontsize=10, color="#E24B4A")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" -> Graphique : {output_path}")
 

def plot_radar_t60(df: pd.DataFrame, output_path: str):
    """
    Graphique radar normalisé pour l'horizon t+60 min (horizon principal).
    On compare les 3 modèles sur RMSE, MAE et R² simultanément.
    Note : RMSE et MAE sont inversés (plus bas = meilleur) pour la lisibilité.
    """
    df_t60 = df[df["horizon"] == "t+60 min"].copy()
    if df_t60.empty:
        return
 
    models = df_t60["model_label"].tolist()
 
    # Normalisation
    rmse_vals = df_t60["rmse_mean"].values
    mae_vals= df_t60["mae_mean"].values
    r2_vals = df_t60["r2_mean"].values
 
    rmse_norm = 1-(rmse_vals - rmse_vals.min())/(rmse_vals.max()- rmse_vals.min() + 1e-9)
    mae_norm = 1-(mae_vals  - mae_vals.min()) /(mae_vals.max()  - mae_vals.min()  + 1e-9)
    r2_norm = (r2_vals-r2_vals.min())/(r2_vals.max() - r2_vals.min() + 1e-9)
 
    categories = ["RMSE\n(↓ meilleur)", "MAE\n(↓ meilleur)", "R²\n(↑ meilleur)"]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
 
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
 
    for i, model in enumerate(models):
        values = [rmse_norm[i], mae_norm[i], r2_norm[i]]
        values += values[:1]
        color = COLORS.get(model, f"C{i}")
        ax.plot(angles, values, "o-", linewidth=2, label=model, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)
 
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title("Comparaison normalisée — t+60 min\n(1 = meilleur modèle sur ce critère)", fontsize=10, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
 
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" -> Radar t+60 : {output_path}")
 
 

# TABLEAU DE SYNTHÈSE ET ANALYSE CRITIQUE
def print_synthesis(df: pd.DataFrame):
    """Affiche le tableau de synthèse et une analyse critique automatique."""
    print(" SYnthèse de la tâche 1: Régression continue")
    for h in HORIZONS:
        h_df = df[df["horizon"] == h].copy()
        if h_df.empty:
            continue
        print(f"\n  Horizon {h} :")
        print(f"{'Modèle':<25} {'RMSE':>12} {'MAE':>12} {'R²':>10}")
 
        for _, row in h_df.iterrows():
            clinical = "✅" if row["rmse_mean"] <= CLINICAL_RMSE_THRESHOLD else "⚠️ "
            print(f" {row['model_label']:<25} "
                  f"{row['rmse_mean']:>6.2f}+/-{row['rmse_std']:<4.2f} "
                  f"{row['mae_mean']:>6.2f}+/-{row['mae_std']:<4.2f} "
                  f"{row['r2_mean']:>8.3f} {clinical}")
 
    # Meilleur modèle sur t+60
    t60 = df[df["horizon"] == "t+60 min"]
    if not t60.empty:
        best_rmse = t60.loc[t60["rmse_mean"].idxmin(), "model_label"]
        best_r2 = t60.loc[t60["r2_mean"].idxmax(),"model_label"]
        print(f"\n Meilleur RMSE à t+60 min : {best_rmse}")
        print(f" Meilleur R² à t+60 min : {best_r2}")
 
        # Analyse de la dégradation par horizon
        print(f"\n  Dégradation RMSE par horizon (meilleur modèle = {best_rmse}) :")
        best_df = df[df["model_label"] == best_rmse]
        for h in HORIZONS:
            row = best_df[best_df["horizon"] == h]
            if not row.empty:
                print(f" {h} : RMSE = {row['rmse_mean'].values[0]:.2f} mg/dL")
 


# Pieplinz 
def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(" COMPARAISON — Tâche 1 : Régression continue")
    df = load_all_results()
    print(f"\n  Modèles chargés : {df['model_label'].unique().tolist()}")
    print(f"Horizons : {df['horizon'].unique().tolist()}")
 
    # Tableau de synthèse
    summary_path= os.path.join(OUTPUT_DIR, "comparison_regression_summary.csv")
    df.to_csv(summary_path, index=False)
    print(f"\n  -> Tableau de synthèse : {summary_path}")
 
    # Graphiques de comparaison
    plot_metric_comparison(df, "rmse_mean", "RMSE (mg/dL)", "rmse_std",os.path.join(OUTPUT_DIR, "compare_rmse.png"))
    plot_metric_comparison(df, "mae_mean",  "MAE (mg/dL)",  "mae_std",os.path.join(OUTPUT_DIR, "compare_mae.png"))
    plot_metric_comparison(df, "r2_mean",   "R²", "r2_std", os.path.join(OUTPUT_DIR, "compare_r2.png"))
    plot_radar_t60(df, os.path.join(OUTPUT_DIR, "radar_t60.png"))
 
    # Synthèse console
    print_synthesis(df)
 
 
if __name__ == "__main__":
    run()
 