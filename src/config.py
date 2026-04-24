"""
config.py

Configuration partagée entre tous les scripts de modélisation.
Importer ce fichier dans chaque script modèle et comparaison.
 
Auteurs : Palliere Raphael — E4 Bio
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

# Chemins 
DATASET_PATH = "data/processed/meal_windows_dataset.csv"
RESULTS_DIR  = "data/results"

# Configuration des cibles et des features
N_FOLDS = 5
RANDOM_STATE = 42
CLINICAL_RMSE_THRESHOLD = 15.0 #Seuil clinique ISO15197 (mg/dL)

#Features de la config A : celle que l'on conserve 
FEATURES_AGG = [
    # CGM pré-repas — statistiques résumées
    "cgm_at_meal",
    "cgm_pre_mean",
    "cgm_pre_std",
    "cgm_pre_min",
    "cgm_pre_max",
    "cgm_slope_15",
    "cgm_slope_30",
    # Nutrition
    "carbs",
    "protein",
    "fat",
    "fiber",
    # Encodage temporel cyclique
    "hour_sin",
    "hour_cos",
    # Biomarqueurs cliniques
    "bio_A1c PDL (Lab)",
    "bio_Fasting GLU - PDL (Lab)",
    "bio_Insulin",
    "bio_BMI",
    "bio_Age",
    "bio_group_encoded",
    "bio_gender_encoded",
]

# Variables catégorielles 
CATEGORICAL_FEATURE = "meal_type"


#Etape 1 : Régression continue 
REGRESSION_TARGETS = {
    "t30": "cgm_target_t30",
    "t60": "cgm_target_t60",   # Horizon principal
    "t90": "cgm_target_t90",
}
 
# Tâche 2 — Classification
CLASSIFICATION_TARGET= "glycemic_label"
 
# Ordre des classes (pour les rapports de classification)
CLASS_ORDER = ["normal", "hyper"]
 
# Seuils cliniques pour labellisation a posteriori (mg/dL)
THRESHOLDS  = {"normal": 140}   # seuil unique : < 140 → normal, ≥ 140 → hyper




def load_dataset(path: str) -> pd.DataFrame:
    """Charge le dataset et vérifie les colonnes essentielles."""
    df = pd.read_csv(path)
    essential = ["patient_id", "glycemic_label", "cgm_target_t30", "cgm_target_t60", "cgm_target_t90"]
    missing = [c for c in essential if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le dataset : {missing}")
    return df


def build_X(df: pd.DataFrame) -> pd.DataFrame:
    # Travailler sur une copie pour ne pas muter le DataFrame original
    df = df.copy()

    # Normalisation de meal_type AVANT tout encodage
    df[CATEGORICAL_FEATURE] = (
        df[CATEGORICAL_FEATURE]
        .str.strip()
        .str.lower()
        .replace({
            "snack 1": "snacks",
            "snack":   "snacks",
            "snacks":  "snacks",   # idempotent, sécurité
        })
    )

    # Construction de X à partir des features agrégées disponibles
    available = [c for c in FEATURES_AGG if c in df.columns]
    X = df[available].copy()

    # One-hot encoding de meal_type sur données déjà normalisées
    if CATEGORICAL_FEATURE in df.columns:
        dummies = pd.get_dummies(df[CATEGORICAL_FEATURE], prefix="meal", drop_first=False)
        X = pd.concat([X.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)

    return X




def get_preprocessing():
    """
    Retourne le pre processeur partagé : imputation médiane + standardisation.
    À intégrer dans chaque Pipeline sklearn.
    Note : le fit se fait UNIQUEMENT sur le train de chaque fold.
    """
    return [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ]
 


def get_cv_splits(X: pd.DataFrame, y: pd.Series, groups: pd.Series):
    """
    Génère les splits GroupKFold.
    Retourne une liste de tuples (train_idx, test_idx).
    """
    gkf = GroupKFold(n_splits=N_FOLDS)
    return list(gkf.split(X, y, groups=groups))
 
 
def label_from_value(v: float) -> str:
    if np.isnan(v):
        return "unknown"
    return "normal" if v < THRESHOLDS["normal"] else "hyper"
 


def save_results(results: list[dict], output_path: str):
    """Sauvegarde une liste de dictionnaires résultats en CSV."""
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"  → Résultats sauvegardés : {output_path}")
 