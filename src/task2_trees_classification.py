"""
task2_trees_classification.py

Classification : prédire l'état glycémique postprandial

Modèles   : Arbre de décision (CART) + Random Forest Classifier

Les deux modèles sont dans le même fichier car leur logique d'évaluation est identique

Nous utilisons class_weight='balanced' pour corriger le biais du déséquilibre des classes à l'entraînement. (cas normaux plus fréquents que les hypoglycémies sévères)
Validation par GroupKFold k=5 strictement par patient (pas de data leakage)


Auteurs : Palliere Raphael & Bouny Mathieu — E4 Bio
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)

from config import (
    DATASET_PATH, RESULTS_DIR, RANDOM_STATE,
    CLASSIFICATION_TARGET, CLASS_ORDER,
    load_dataset, build_X, get_preprocessing, get_cv_splits, save_results,
    label_from_value,
)
 
warnings.filterwarnings("ignore")
 
# Horizons
HORIZON_TARGETS = {
    "t30": "cgm_target_t30",
    "t60": "cgm_target_t60",
    "t90": "cgm_target_t90",
}

#Hyperparamètres arbre de décision 
DT_PARAMS ={
    "max_depth": 5, 
    "min_samples_leaf": 5,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
}

# Hyperparamètres Random Forest
RF_PARAMS = {
    "n_estimators":  200,
    "max_depth":  8,
    "min_samples_leaf":5,
    "max_features": "sqrt",
    "class_weight":  "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

#FOnctions utilisés pour les deux modèles : 
def build_labels(df: pd.DataFrame, target_col: str) -> pd.Series:
    return df[target_col].apply(label_from_value)
 
def evaluate_fold(pipeline, X_train, y_train, X_test, y_test):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
 
    mask = y_test != "unknown"
    y_test_c = y_test[mask]
    y_pred_c = y_pred[mask]
    classes  = [c for c in CLASS_ORDER if c in y_test_c.values]
 
    return {
        "accuracy": float(accuracy_score(y_test_c, y_pred_c)),
        "recall":   float(recall_score(
            y_test_c,
            y_pred_c,
            average="macro",
            zero_division=0,
            labels=classes)),
        "f1": float(f1_score(y_test_c, y_pred_c, average="macro", zero_division=0, labels=classes)),
        "y_test":   y_test_c.values,
        "y_pred":   y_pred_c,
    }


def plot_confusion_matrix(y_test, y_pred, title: str, output_path: str):
    labels = [c for c in CLASS_ORDER if c in np.unique(np.concatenate([y_test, y_pred]))]
    cm = confusion_matrix(y_test, y_pred, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        ax=ax, colorbar=True, cmap="Blues", values_format=".2f"
    )
    ax.set_title(title, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
 
def plot_feature_importance(pipeline: Pipeline, feature_names: list, title: str, output_path: str):
    importances = pipeline.named_steps["model"].feature_importances_
    idx = np.argsort(importances)[::-1][:20]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh([feature_names[i] for i in idx][::-1], importances[idx][::-1], color="#3B8BD4", alpha=0.85)
    ax.set_xlabel("Importance (réduction d'impureté)", fontsize=10)
    ax.set_title(title, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
 

def plot_decision_tree_classifier(pipeline: Pipeline, feature_names: list, class_names: list, output_path: str):
    tree = pipeline.named_steps["model"]
    fig, ax = plt.subplots(figsize=(20, 8))
    plot_tree(tree, feature_names=feature_names, class_names=class_names, filled=True, rounded=True, max_depth=3, fontsize=8, ax=ax, impurity=False, precision=2)
    ax.set_title("Arbre de décision — Classification glycémique (3 niveaux)\nt+60 min", fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f" -> Arbre visualisé : {output_path}")
 

#Evaluation d'un modèle sur 3 horizons 
def evaluate_model(model_name: str, pipeline_fn, output_dir: str, df: pd.DataFrame, X: pd.DataFrame, groups: pd.Series) -> list:
    """
    Évalue un modèle de classification sur les 3 horizons.
    pipeline_fn : callable qui retourne un Pipeline sklearn.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_results = []
 
    print(f"\n {model_name} :")
 
    for horizon_key, target_col in HORIZON_TARGETS.items():
        y = build_labels(df, target_col)
        valid = (y != "unknown") & df[target_col].notna()
        X_v = X[valid].reset_index(drop=True)
        y_v = y[valid].reset_index(drop=True)
        g_v = groups[valid].reset_index(drop=True)
 
        print(f"\n Horizon {horizon_key} — {valid.sum()} fenêtres valides")
        print(f" Distribution : {y_v.value_counts().to_dict()}")
 
        pipeline = pipeline_fn()
        splits   = get_cv_splits(X_v, y_v, g_v)
 
        accs, recalls, f1s = [], [], []
        all_y_test, all_y_pred = [], []
 
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            res = evaluate_fold(
                pipeline,
                X_v.iloc[train_idx], y_v.iloc[train_idx],
                X_v.iloc[test_idx],  y_v.iloc[test_idx],
            )
            accs.append(res["accuracy"])
            recalls.append(res["recall"])
            f1s.append(res["f1"])
            all_y_test.extend(res["y_test"])
            all_y_pred.extend(res["y_pred"])
 
            print(f" Fold {fold_idx+1} | Acc={res['accuracy']:.3f} | "f"Recall={res['recall']:.3f} | F1={res['f1']:.3f}")
 
        acc_m, acc_s= np.mean(accs), np.std(accs)
        recall_m, recall_s = np.mean(recalls),np.std(recalls)
        f1_m, f1_s = np.mean(f1s),np.std(f1s)
 
        print(f"\n  -> Acc={acc_m:.3f}±{acc_s:.3f} | "f"Recall={recall_m:.3f}±{recall_s:.3f} | F1={f1_m:.3f}±{f1_s:.3f}")
 
        y_test_all = np.array(all_y_test)
        y_pred_all = np.array(all_y_pred)
        classes_all = [c for c in CLASS_ORDER if c in np.unique(y_test_all)]
 
        # Rapport détaillé
        report = classification_report(
            y_test_all, y_pred_all, labels=classes_all,
            output_dict=True, zero_division=0,
        )
        pd.DataFrame(report).T.to_csv(
            os.path.join(output_dir, f"report_{horizon_key}.csv")
        )
 
        # Matrice de confusion
        plot_confusion_matrix(
            y_test_all, y_pred_all,
            title=f"{model_name} — t+{horizon_key[1:]} min",
            output_path=os.path.join(output_dir, f"confusion_{horizon_key}.png"),
        )
 
        # Importance features
        plot_feature_importance(
            pipeline, list(X_v.columns),
            title=f"{model_name} — Importance features | t+{horizon_key[1:]} min",
            output_path=os.path.join(output_dir, f"importance_{horizon_key}.png"),
        )
 
        # Visualisation arbre pour t+60 (Decision Tree uniquement)
        if horizon_key == "t60" and model_name == "DecisionTree":
            plot_decision_tree_classifier(
                pipeline, list(X_v.columns), classes_all,
                output_path=os.path.join(output_dir, "tree_structure_t60.png"),
            )
 
        all_results.append({
            "model":          model_name,
            "task":           "classification",
            "horizon":        f"t+{horizon_key[1:]} min",
            "accuracy_mean":  round(acc_m, 3),
            "accuracy_std":   round(acc_s, 3),
            "recall_mean":    round(recall_m, 3),
            "recall_std":     round(recall_s, 3),
            "f1_mean":        round(f1_m, 3),
            "f1_std":         round(f1_s, 3),
            "n_features":     X_v.shape[1],
            "class_weight":   "balanced",
        })
 
    return all_results
 
#Pipeline principal
def run():
    print("TÂCHE 2 — Classification | Arbres de décision + Random Forest")
    df = load_dataset(DATASET_PATH)
    X= build_X(df)
    groups = df["patient_id"]
 
    print(f"\n Dataset : {len(df)} fenêtres | {df['patient_id'].nunique()} patients")
    print(f" Features : {X.shape[1]} colonnes (Config A)")
 
    all_results = []
 
    #Arbre de décision 
    dt_dir = os.path.join(RESULTS_DIR, "task2_decision_tree")
    dt_results = evaluate_model(
        model_name="DecisionTree",
        pipeline_fn=lambda: Pipeline(
            get_preprocessing() + [("model", DecisionTreeClassifier(**DT_PARAMS))]
        ),
        output_dir=dt_dir,
        df=df, X=X, groups=groups,
    )
    save_results(dt_results, os.path.join(dt_dir, "results_decision_tree_classification.csv"))
    all_results.extend(dt_results)
 
    # Random Forest 
    rf_dir = os.path.join(RESULTS_DIR, "task2_random_forest")
    rf_results = evaluate_model(
        model_name="RandomForest",
        pipeline_fn=lambda: Pipeline(get_preprocessing() + [("model", RandomForestClassifier(**RF_PARAMS))]),
        output_dir=rf_dir,
        df=df, X=X, groups=groups,
    )
    save_results(rf_results, os.path.join(rf_dir, "results_random_forest_classification.csv"))
    all_results.extend(rf_results)
 
    # Résumé global des deux modèles
    save_results(
        all_results,
        os.path.join(RESULTS_DIR, "task2_trees_classification_summary.csv"),
    )
 
 
if __name__ == "__main__":
    run()
