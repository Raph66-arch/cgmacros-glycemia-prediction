"""
compare_task2_classification.py
Comparaison des résultats de la Tâche 2 — Classification glycémique.

Le code produit : 
- Un tableau de synthèse complet (CSV)
- Des graphiques de comparaison (Accuracy, Recall, F1) par horizon
- Une analyse des erreurs cliniquement critiques (faux négatifs hypoglycémie)


Auteurs : Palliere Raphael & Bouny Mathieu — E4 Bio
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
from config import RESULTS_DIR
 
warnings.filterwarnings("ignore")
 
OUTPUT_DIR = os.path.join(RESULTS_DIR, "comparison_task2_classification")
 
RESULT_FILES = {
    "LogisticRegression": os.path.join(RESULTS_DIR, "task2_logistic_regression", "results_logistic_regression_classification.csv"),
    "DecisionTree":       os.path.join(RESULTS_DIR, "task2_decision_tree", "results_decision_tree_classification.csv"),
    "RandomForest":       os.path.join(RESULTS_DIR, "task2_random_forest", "results_random_forest_classification.csv"),
}
 
HORIZONS = ["t+30 min", "t+60 min", "t+90 min"]
COLORS   = {
    "LogisticRegression": "#EB4C4C",
    "DecisionTree":       "#3B8BD4",
    "RandomForest":       "#1D9E75",
}
 

# Fonction pour charger tous les résultats et les combiner en un seul DataFrame
def load_all_results() -> pd.DataFrame:
    frames = []
    for model_name, path in RESULT_FILES.items():
        df = pd.read_csv(path)
        df["model_label"] = model_name
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    if "horizon" not in combined.columns and "Horizon" in combined.columns:
        combined = combined.rename(columns={"Horizon": "horizon"})
    return combined


#Visualisations : 
def plot_metric_comparison(df: pd.DataFrame, metric: str, metric_label: str, std_col: str, output_path: str):
    """Barplot groupé par horizon pour une métrique donnée."""
    models= df["model_label"].unique().tolist()
    horizons = [h for h in HORIZONS if h in df["horizon"].values]

    x  = np.arange(len(horizons))
    width = 0.25
    n = len(models)
    offsets = np.linspace(-(n - 1) * width / 2, (n - 1) * width / 2, n)
 
    fig, ax = plt.subplots(figsize=(9, 5))
 
    for i, model in enumerate(models):
        model_df = df[df["model_label"] == model]
        values, stds = [], []
        for h in horizons:
            row = model_df[model_df["horizon"] == h]
            values.append(float(row[metric].values[0]) if len(row) > 0 else 0)
            stds.append(float(row[std_col].values[0]) if len(row) > 0 else 0)
 
        ax.bar(x + offsets[i], values, width, yerr=stds, capsize=4, label=model, color=COLORS.get(model, f"C{i}"), alpha=0.85)
    # Ligne de référence à 0.8 (seuil de qualité indicatif pour la classification)
    ax.axhline(0.8, color="gray", linestyle=":", linewidth=0.9, label="Référence 0.80", alpha=0.7)
 
    ax.set_xticks(x)
    ax.set_xticklabels(horizons)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel(metric_label, fontsize=11)
    ax.set_title(f"Tâche 2 — Classification glycémique\n{metric_label} par modèle et horizon", fontsize=11)
    ax.legend(fontsize=9)
 
    # Annoter le meilleur par horizon
    for j, h in enumerate(horizons):
        h_df = df[df["horizon"] == h]
        if not h_df.empty:
            best_val = h_df[metric].max()
            ax.text(j, best_val + 0.02, "★", ha="center", fontsize=10, color="#E24B4A")
 
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Graphique : {output_path}")
 


def plot_f1_heatmap(df: pd.DataFrame, output_path: str):
    """
    Heatmap F1-score : lignes = modèles, colonnes = horizons.
    Vue d'ensemble rapide des performances de classification.
    """
    models = df["model_label"].unique().tolist()
    horizons = [h for h in HORIZONS if h in df["horizon"].values]
 
    matrix = np.zeros((len(models), len(horizons)))
    for i, model in enumerate(models):
        for j, h in enumerate(horizons):
            row = df[(df["model_label"] == model) & (df["horizon"] == h)]
            if not row.empty:
                matrix[i, j] = row["f1_mean"].values[0]
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
 
    ax.set_xticks(range(len(horizons)))
    ax.set_yticks(range(len(models)))
    ax.set_xticklabels(horizons, fontsize=10)
    ax.set_yticklabels(models, fontsize=10)
 
    for i in range(len(models)):
        for j in range(len(horizons)):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=11, color="black", fontweight="bold")
 
    plt.colorbar(im, ax=ax, label="F1-score macro")
    ax.set_title("Heatmap F1-score — Tâche 2 Classification\n(macro, GroupKFold k=5)", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  → Heatmap F1 : {output_path}")
 

def plot_recall_focus(df: pd.DataFrame, output_path: str):
    """
    Focus sur le Recall macro à t+60 min.
    Le recall est la métrique clinique la plus critique :
    un faux négatif sur 'hypo' (manquer une hypoglycémie) est plus grave
    qu'un faux positif (fausse alarme).
    """
    df_t60 = df[df["horizon"] == "t+60 min"].copy()
    if df_t60.empty:
        return
 
    models  = df_t60["model_label"].tolist()
    recalls = df_t60["recall_mean"].values
    stds    = df_t60["recall_std"].values
    colors  = [COLORS.get(m, "gray") for m in models]
 
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(models, recalls, yerr=stds, capsize=5, color=colors, alpha=0.85, width=0.5)
 
    # Annoter les valeurs
    for bar, val in zip(bars, recalls):
        ax.text(bar.get_x() + bar.get_width() / 2,bar.get_height() + 0.02, f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.axhline(0.8, color="gray", linestyle=":", linewidth=0.9, alpha=0.7, label="Référence 0.80")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Recall macro", fontsize=11)
    ax.set_title("Recall macro à t+60 min — Focus clinique\n" "(rappel : un faux négatif hyperglycémie sévère est cliniquement critique)", fontsize=10)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Focus recall : {output_path}")
 
 

#Tableau de synthèse
def print_synthesis(df: pd.DataFrame):
    print("SYNTHÈSE — TÂCHE 2 : Classification glycémique")
    for h in HORIZONS:
        h_df = df[df["horizon"] == h].copy()
        if h_df.empty:
            continue
        print(f"\n  Horizon {h} :")
        print(f"  {'Modèle':<25} {'Accuracy':>12} {'Recall':>12} {'F1':>12}")
 
        for _, row in h_df.iterrows():
            print(f"  {row['model_label']:<25} "
                  f"{row['accuracy_mean']:>6.3f}±{row['accuracy_std']:<4.3f} "
                  f"{row['recall_mean']:>6.3f}±{row['recall_std']:<4.3f} "
                  f"{row['f1_mean']:>6.3f}±{row['f1_std']:<4.3f}")
 
    # Meilleur modèle sur t+60
    t60 = df[df["horizon"]=="t+60 min"]
    if not t60.empty:
        best_f1 = t60.loc[t60["f1_mean"].idxmax(),     "model_label"]
        best_recall= t60.loc[t60["recall_mean"].idxmax(), "model_label"]
        print(f"\nMeilleur F1-score macro à t+60 min : {best_f1}")
        print(f"Meilleur Recall macro à t+60 min : {best_recall}")
 
        print(f"\n  Note clinique :")
        print(f" Le Recall est la métrique prioritaire pour la détection des hyperglycémies — " f"un faux négatif (hyperglycémie sévère manquée) est plus grave cliniquement qu'un faux positif " f"(fausse alarme). Vérifier le recall par classe 'hyper_severe' dans les rapports détaillés.")
 
#Pipeline de comparaison
def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("COMPARAISON — Tâche 2 : Classification glycémique")
 
    df = load_all_results()
    print(f"\n  Modèles chargés: {df['model_label'].unique().tolist()}")
    print(f"  Horizons :{df['horizon'].unique().tolist()}")
 
    # Sauvegarde tableau de synthèse
    summary_path = os.path.join(OUTPUT_DIR, "comparison_classification_summary.csv")
    df.to_csv(summary_path, index=False)
    print(f"\n  → Tableau de synthèse : {summary_path}")
 
    # Graphiques
    plot_metric_comparison(df, "accuracy_mean","Accuracy",   "accuracy_std", os.path.join(OUTPUT_DIR, "compare_accuracy.png"))
    plot_metric_comparison(df, "recall_mean", "Recall macro", "recall_std", os.path.join(OUTPUT_DIR, "compare_recall.png"))
    plot_metric_comparison(df, "f1_mean", "F1-score macro", "f1_std",os.path.join(OUTPUT_DIR, "compare_f1.png"))
    plot_f1_heatmap(df, os.path.join(OUTPUT_DIR, "heatmap_f1.png"))
    plot_recall_focus(df,os.path.join(OUTPUT_DIR, "focus_recall_t60.png"))
 
    # Synthèse console
    print_synthesis(df)
 
 
if __name__ == "__main__":
    run()




