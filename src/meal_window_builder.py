"""
meal_window_builder.py
======================
Pipeline de construction des fenêtres repas pour le projet CGMacros.
 
Ce script extrait, pour chaque repas horodaté de chaque patient :
  - La séquence CGM pré-repas (contexte temporel, input du modèle)
  - La valeur glycémique de t0 à t+90min (variable à prédire)
  - Les macronutriments du repas (features statiques repas)
  - Les biomarqueurs cliniques du patient (features statiques patient)
 
Sortie : un fichier CSV unique `meal_windows_dataset.csv` prêt pour la modélisation.
 
 
Auteur : Palliere Raphael — E4 Bio
"""


import os
import glob
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
 
warnings.filterwarnings("ignore")


# Configuration général
CONFIG = {
    # Chemins
    "data_raw_dir":       "data/raw",
    "bio_path":           "data/processed/bio_with_group.csv",
    "output_dir":         "data/processed",
    "output_filename":    "meal_windows_dataset.csv",
 
    # Paramètres de la fenêtre temporelle (en minutes)
    "pre_meal_window": 60,   # Durée de la séquence CGM avant le repas
    "post_meal_target": 90,   # On extrait jusqu'à t+90 min
    "post_meal_horizons": [30, 60, 90],  # Les 3 horizons de prédiction
 
    # Capteur CGM principal
    "cgm_column":         "Libre GL",   # Abbott FreeStyle Libre — présent chez tous
 
    # Résolution temporelle attendue après interpolation (minutes)
    "cgm_resolution":     1,

    # Numéros de patients valides (24, 25, 37, 40 n'existent pas, numéros jusqu'à 49)
    "valid_patients": [i for i in range(1, 50) if i not in [12, 24, 25, 37, 40]],
 
    # Seuils cliniques pour la cible (mg/dL) — utilisés pour labellisation a posteriori
    "thresholds": {
        "hypo":   70,    # < 70 → hypoglycémie
        "normal": 140,   # 70–140 → normoglycémie postprandiale
        "hyper":  180,   # 140–180 → hyperglycémie légère, >180 → sévère
    },
 
    # Variables biomarqueurs cliniques à conserver de bio_with_group.csv
    "bio_features": [
        "Age", "BMI",
        "A1c PDL (Lab)",          # HbA1c
        "Fasting GLU - PDL (Lab)", # Glycémie à jeun
        "Insulin",                 # Insuline à jeun
        "Triglycerides",
        "HDL",
    ],
 
    # Variables biomarqueurs à exclure (colinéarité forte)
    "bio_exclude": [
        "Cholesterol",   # r=0.88 avec LDL
        "LDL (Cal)",     # colinéaire avec Cholesterol
        "VLDL (Cal)",    # colinéaire avec Triglycérides
        "Non HDL",       # dérivé de Cholesterol - HDL
        "Cho/HDL Ratio", # dérivé
        "Body weight",   # redondant avec BMI
        "Height",        # redondant avec BMI
    ],
 
    # Colonnes de macronutriments attendues dans les fichiers patient
    "meal_features": ["carbs", "protein", "fat", "fiber", "calories"],
 
    # Nombre minimum de points CGM valides requis dans la fenêtre pré-repas
    "min_valid_cgm_points": 30,  # Sur 60 points à 1 min → tolérance de 50% de manquants
}
 



 # Téléchargement des données
def load_bio(bio_path: str) -> pd.DataFrame:
    """
    Charge le fichier bio_with_group.csv et sélectionne les colonnes pertinentes.
    Retourne un DataFrame indexé par subject.
    """
    bio = pd.read_csv(bio_path)
 
    # Nettoyer les noms de colonnes (surtout les espaces parasites)
    bio.columns = bio.columns.str.strip()
 
    #Vérifier la présence de la colonne group
    if "group" not in bio.columns:
        raise ValueError(
            "Colonne 'group' absente de bio_with_group.csv. "
            "Lancer d'abord build_patient_table.py."
        )
 
    #Encoder le groupe métabolique en chiffre. Cela nous sera plus simple pour plus tard pour sélectionner les groupes. 
    group_map = {"healthy": 0, "prediabetes": 1, "t2d": 2}
    bio["group_encoded"] = bio["group"].map(group_map)
 
    #Encoder le genre (juste avec un 0, Masculin et 1 féminin. 
    bio["gender_encoded"] = bio["Gender"].map({"M": 0, "F": 1})
 
    # Sélectionner des colonnes
    keep = ["subject", "group", "group_encoded", "gender_encoded"] + CONFIG["bio_features"]
    available = [c for c in keep if c in bio.columns]
    missing = [c for c in keep if c not in bio.columns]
    if missing:
        print(f"⚠️  Colonnes bio absentes (ignorées) : {missing}")
 
    bio = bio[available].copy()
    bio = bio.set_index("subject")
 
    return bio
 
 
def load_patient_csv(filepath: str) -> pd.DataFrame:
    """
    Charge le fichier csv d'un patient CGMacros-0XX.xlsx.
    Normalise les noms de colonnes, parse les timestamps.
    Retourne un DataFrame trié par temps.
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
 
    # Parser le timestamp (format nord-américain MM/DD/YYYY HH:MM)
    timestamp_col = _find_column(df, ["Timestamp", "timestamp", "Time", "DATE"])
    if timestamp_col is None:
        raise ValueError(f"Colonne Timestamp introuvable dans {filepath}")
 
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], format="mixed", dayfirst=False)
    df = df.rename(columns={timestamp_col: "timestamp"})
    df = df.sort_values("timestamp").reset_index(drop=True)
 
    return df
 
 
def _find_column(df: pd.DataFrame, candidates: list) -> str | None:
    """Cherche la première colonne dont le nom correspond à l'un des candidats."""
    for c in candidates:
        for col in df.columns:
            if c.lower() in col.lower():
                return col
    return None
 

 # Détection des repas
def detect_meals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nous identifions les événements repas dans le fichier patient.
    Un repas = une ligne où 'Meal Type' est renseigné (non-NaN, non-vide).
    """
    meal_type_col = _find_column(df, ["Meal Type", "meal_type", "MealType"])

    if meal_type_col is None:
        print(" Colonne 'Meal Type' non trouvée — repas non détectables.")
        return pd.DataFrame()

    #On filtre uniquement les lignes avec un type de repas renseigné
    mask = df[meal_type_col].notna() & (df[meal_type_col].astype(str).str.strip() != "")
    meals = df[mask].copy()

    if meals.empty:
        return pd.DataFrame()

    #On reprend exactement les noms de colonnes CGMacros
    rename_map = {meal_type_col: "meal_type"}
    for src, dst in [("Calories", "calories"), ("Carbs", "carbs"),
                     ("Protein", "protein"), ("Fat", "fat"), ("Fiber", "fiber")]:
        col = _find_column(df, [src])
        # Attention : éviter de capturer 'Calories (Activity)'
        if col and "(Activity)" not in col and "(activity)" not in col:
            rename_map[col] = dst

    meals = meals.rename(columns=rename_map)

    for col in ["carbs", "protein", "fat", "fiber", "calories", "meal_type"]:
        if col not in meals.columns:
            meals[col] = np.nan

    return meals[["timestamp", "carbs", "protein", "fat",
                  "fiber", "calories", "meal_type"]].reset_index(drop=True)


 # Construction du type de la colonne repas 
def build_meal_window(
    cgm_series: pd.Series,
    meal_time: pd.Timestamp,
    pre_minutes: int,
    target_minutes: int,
    min_valid_points: int,
) -> dict | None:
    """
    Construit la fenêtre temporelle autour d'un repas.
    Paramètres :
    ----------
    cgm_series   : Series indexée par timestamp, valeurs en mg/dL (Abbott)
    meal_time    : timestamp du début du repas
    pre_minutes  : durée de la fenêtre pré-repas (minutes)
    target_minutes : horizon de prédiction (minutes après le repas)
    min_valid_points : nombre minimum de points valides requis
 
    Retourne :
    --------
    dict avec :
      - cgm_pre_XX : valeurs CGM à chaque minute de la fenêtre pré-repas
      - cgm_at_meal : glycémie au moment du repas (t=0)
      - cgm_target : glycémie à t+target_minutes (variable cible)
      - cgm_pre_mean, _std, _min, _max, _slope_15, _slope_30 : features agrégées
    ou None si la fenêtre est invalide.
    """
    t_start  = meal_time - pd.Timedelta(minutes=pre_minutes)
    t_end    = meal_time + pd.Timedelta(minutes=target_minutes)
 
    # Extraire les points dans la fenêtre complète
    window = cgm_series[(cgm_series.index >= t_start) & (cgm_series.index <= t_end)]
 
    if window.empty:
        return None
 
    # --- Séquence pré-repas ---
    pre_window = cgm_series[
        (cgm_series.index >= t_start) & (cgm_series.index < meal_time)
    ]

    #Mis par sécurité, car je n'ai pas vérifié ligne par ligne mais normalement, nous ne sommes dans un aucun cas ici
    if pre_window.dropna().shape[0] < min_valid_points:
        return None  # Trop de données manquantes dans la fenêtre pré-repas
 



    # Valeurs cibles aux 3 horizons + séquence complète post-repas
    targets = {}
    for h in CONFIG["post_meal_horizons"]:
        t_h = meal_time + pd.Timedelta(minutes=h)
        h_window = cgm_series[
            (cgm_series.index >= t_h - pd.Timedelta(minutes=2)) &
            (cgm_series.index <= t_h + pd.Timedelta(minutes=5))
        ]
        targets[f"cgm_target_t{h}"] = float(h_window.dropna().iloc[0]) if not h_window.dropna().empty else np.nan

    # Vérifier qu'au moins t+60 est disponible (horizon principal)
    if np.isnan(targets.get("cgm_target_t60", np.nan)):
        return None

    # Séquence post-repas minute par minute jusqu'à t+90 (pour exploration future, si vous avons le temps)
    t_post_end = meal_time + pd.Timedelta(minutes=CONFIG["post_meal_target"])
    post_reindexed = cgm_series[
        (cgm_series.index > meal_time) & (cgm_series.index <= t_post_end)
    ].reindex(
        pd.date_range(
            start=meal_time + pd.Timedelta(minutes=1),
            end=t_post_end,
            freq="1min"
        )
    )
    for i, val in enumerate(post_reindexed.values, start=1):
        targets[f"cgm_post_t{i}"] = float(val) if not np.isnan(val) else np.nan


 
    # Valeur au moment du repas :
    meal_window_exact = cgm_series[
        (cgm_series.index >= meal_time - pd.Timedelta(minutes=2)) &
        (cgm_series.index <= meal_time + pd.Timedelta(minutes=2))
    ]
    cgm_at_meal = meal_window_exact.dropna().mean() if not meal_window_exact.dropna().empty else np.nan
 
    #Features agrégées sur la séquence pré-repas 
    pre_values = pre_window.dropna().values
 
    #Pente sur les 15 dernières minutes avant le repas
    slope_15 = _compute_slope(cgm_series, meal_time, lookback_minutes=15)
    # Pente sur les 30 dernières minutes avant le repas
    slope_30 = _compute_slope(cgm_series, meal_time, lookback_minutes=30)
 
    result = {
        "cgm_at_meal":    cgm_at_meal,
        **targets,
        "cgm_pre_mean":   float(np.nanmean(pre_values)),
        "cgm_pre_std":    float(np.nanstd(pre_values)),
        "cgm_pre_min":    float(np.nanmin(pre_values)),
        "cgm_pre_max":    float(np.nanmax(pre_values)),
        "cgm_slope_15":   slope_15,
        "cgm_slope_30":   slope_30,
        "n_valid_pre":    int(pre_window.dropna().shape[0]),
    }
 
    # Séquence brute pré-repas (pour le modèle LSTM/GRU)
    # Rééchantillonnée à 1 min sur pre_minutes points, NaN si manquant
    pre_reindexed = pre_window.reindex(
        pd.date_range(start=t_start, end=meal_time - pd.Timedelta(minutes=1), freq="1min")
    )
    for i, val in enumerate(pre_reindexed.values):
        result[f"cgm_t-{pre_minutes - i}"] = float(val) if not np.isnan(val) else np.nan
 
    return result
 
 
def _compute_slope(cgm_series: pd.Series, meal_time: pd.Timestamp, lookback_minutes: int) -> float:
    """
    Calcule la pente linéaire de la glycémie sur les `lookback_minutes`
    minutes précédant meal_time. Retourne np.nan si insuffisant.
    """
    t_start = meal_time - pd.Timedelta(minutes=lookback_minutes)
    segment = cgm_series[(cgm_series.index >= t_start) & (cgm_series.index < meal_time)].dropna()
 
    if segment.shape[0] < 5:
        return np.nan
 
    x = np.arange(len(segment))
    try:
        slope = float(np.polyfit(x, segment.values, 1)[0])
    except Exception:
        slope = np.nan
 
    return slope


# Encodage temporel cyclique (c'est un plus que nous a donné l'ia, ça va peut-être servir pour les horaires proches mais peut-être pas pour les jours honnêtement)
def encode_meal_time(meal_time: pd.Timestamp) -> dict:
    """
    Encode l'heure du repas de façon cyclique (sin/cos) pour capturer
    la continuité temporelle (23h et 0h sont proches).
    Encode aussi le jour de la semaine (lund-dim).
    """
    hour = meal_time.hour + meal_time.minute / 60.0
    dow  = meal_time.dayofweek  # 0=lundi, 6=dimanche
 
    return {
        "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
        "hour_cos": float(np.cos(2 * np.pi * hour / 24)),
        "dow_sin":  float(np.sin(2 * np.pi * dow / 7)),
        "dow_cos":  float(np.cos(2 * np.pi * dow / 7)),
    }



# Labelisation du niveau de glycémie postprandiale
def label_glycemic_state(cgm_value: float) -> str:
    """
    Classe la valeur glycémique postprandiale selon les seuils cliniques.
    Utilisé pour la tâche de classification secondaire.
    """
    if np.isnan(cgm_value):
        return "unknown"
    thresholds = CONFIG["thresholds"]
    if cgm_value < thresholds["hypo"]:
        return "hypo"
    elif cgm_value < thresholds["normal"]:
        return "normal"
    elif cgm_value < thresholds["hyper"]:
        return "hyper_mild"
    else:
        return "hyper_severe"
    



# Pipeline principal (ce qui est dit dans l'intro de ce code)
def build_meal_windows_dataset(
    data_raw_dir: str,
    bio_path: str,
    output_dir: str,
    output_filename: str,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Pipeline complet : parcourt tous les fichiers patient, construit les
    fenêtres repas, fusionne avec les données bio, et sauvegarde le dataset.
 
    Retourne le DataFrame final fdans un fichier csv. 
    """
    def log(msg):
        if verbose:
            print(msg)
 

    log("  PIPELINE — Construction des fenêtres repas (CGMacros)")
 
    # 1. Charger les données bio 
    log("\n[1/4] Chargement du profil patients...")
    bio = load_bio(bio_path)
    log(f"  → {len(bio)} patients chargés | groupes : {bio['group'].value_counts().to_dict()}")
 
    # 2. Lister les fichiers patient
    # Nouvelle structure : data/raw/CGMacros-0XX/CGMacros-0XX.csv
    patient_files = []
    for pid in CONFIG["valid_patients"]:
        folder = os.path.join(data_raw_dir, f"CGMacros-{pid:03d}")
        csv_file = os.path.join(folder, f"CGMacros-{pid:03d}.csv")
        if os.path.exists(csv_file):
            patient_files.append((pid, csv_file))
        else:
            print(f"⚠️  Fichier manquant : {csv_file}")
    
 
    if not patient_files:
        raise FileNotFoundError(
            f"Aucun fichier CGMacros-*.xlsx trouvé dans '{data_raw_dir}'.\n"
            f"Vérifiez que les données brutes sont bien placées dans ce dossier."
        )
 
    log(f"\n[2/4] Fichiers patients trouvés : {len(patient_files)}")
 
    # 3. Construire les fenêtres repas 
    log(f"\n[3/4] Construction des fenêtres repas...")
    log(f"  Fenêtre pré-repas : {CONFIG['pre_meal_window']} min")
    log(f"  Horizon prédiction : t+{CONFIG['post_meal_target']} min")
    log(f"  Capteur CGM : {CONFIG['cgm_column']}")
 
    all_windows = []
    stats = {"total_meals": 0, "valid_windows": 0, "skipped_no_cgm": 0, "skipped_invalid": 0}
 
    for patient_id, filepath in patient_files:
        # Extraire l'identifiant patient depuis le nom de fichier
        log(f"\n  → Patient {patient_id:03d} ({filepath})")
 
        try:
            df = load_patient_csv(filepath)
        except Exception as e:
            log(f"❌ Erreur chargement : {e}")
            continue
 
        #Construire la série CGM indexée par timestamp
        cgm_col = _find_column(df, [CONFIG["cgm_column"], "Libre", "Abbott"])
        if cgm_col is None:
            log(f"⚠️  Colonne CGM '{CONFIG['cgm_column']}' non trouvée — patient ignoré")
            stats["skipped_no_cgm"] += 1
            continue
 
        cgm_series = df.set_index("timestamp")[cgm_col].astype(float)
        cgm_series = cgm_series[~cgm_series.index.duplicated(keep="first")]
 
        #Détecter les repas
        meals = detect_meals(df)
        if meals.empty:
            log(f"⚠️  Aucun repas détecté")
            continue
 
        log(f" {len(meals)} repas détectés")
        stats["total_meals"] += len(meals)
 
        # Données bio du patient
        if patient_id not in bio.index:
            log(f"⚠️  Patient {patient_id} absent de bio_with_group.csv — ignoré")
            continue
        patient_bio = bio.loc[patient_id].to_dict()
 
        #Construire une fenêtre par repas
        for _, meal in meals.iterrows():
            window = build_meal_window(
                cgm_series=cgm_series,
                meal_time=meal["timestamp"],
                pre_minutes=CONFIG["pre_meal_window"],
                target_minutes=CONFIG["post_meal_target"],
                min_valid_points=CONFIG["min_valid_cgm_points"],
            )
 
            if window is None:
                stats["skipped_invalid"] += 1
                continue
 
            #Fusionner toutes les informations
            row = {
                "patient_id":  patient_id,
                "meal_time":   meal["timestamp"],
                **encode_meal_time(meal["timestamp"]),  # Encodage cyclique heure
                # Macronutriments
                "carbs":       meal.get("carbs", np.nan),
                "protein":     meal.get("protein", np.nan),
                "fat":         meal.get("fat", np.nan),
                "fiber":       meal.get("fiber", np.nan),
                "calories":    meal.get("calories", np.nan),
                "meal_type":   meal.get("meal_type", np.nan),
                # Fenêtre CGM (séquence + agrégats)
                **window,
                # Biomarqueurs patient (répétés à chaque fenêtre)
                **{f"bio_{k}": v for k, v in patient_bio.items()},
                # Labellisation clinique a posteriori
                "glycemic_label": label_glycemic_state(window.get("cgm_target_t60", np.nan)),
            }
 
            all_windows.append(row)
            stats["valid_windows"] += 1
 
    # 4. Assembler et sauvegarder
    log(f"\n[4/4] Assemblage du dataset final...")
 
    if not all_windows:
        raise ValueError("Aucune fenêtre valide construite. Vérifier les données brutes.")
 
    dataset = pd.DataFrame(all_windows)
 
    # Réorganiser les colonnes : identifiant → temporel → nutrition → CGM agrégé → CGM séquence → bio → cible
    priority_cols = [
        "patient_id", "meal_time", "meal_type",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "carbs", "protein", "fat", "fiber", "calories",
        "cgm_at_meal", "cgm_pre_mean", "cgm_pre_std",
        "cgm_pre_min", "cgm_pre_max", "cgm_slope_15", "cgm_slope_30",
        "n_valid_pre",
    ]
    bio_cols   = [c for c in dataset.columns if c.startswith("bio_")]
    seq_cols   = sorted([c for c in dataset.columns if c.startswith("cgm_t-")],
                        key=lambda x: int(x.split("-")[1]), reverse=True)
    target_col = ["cgm_target", "glycemic_label"]
 
    ordered_cols = (
        [c for c in priority_cols if c in dataset.columns]
        + bio_cols
        + seq_cols
        + [c for c in target_col if c in dataset.columns]
    )
    remaining = [c for c in dataset.columns if c not in ordered_cols]
    dataset = dataset[ordered_cols + remaining]
 
    # Sauvegarder
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    dataset.to_csv(output_path, index=False)
 
    # --- Rapport final ---
    log("  RÉSUMÉ")
    log(f"  Repas détectés total    : {stats['total_meals']}")
    log(f"  Fenêtres valides        : {stats['valid_windows']}")
    log(f"  Ignorées (pas de CGM)   : {stats['skipped_no_cgm']} patients")
    log(f"  Ignorées (invalides)    : {stats['skipped_invalid']} fenêtres")
    log(f"  Colonnes générées       : {len(dataset.columns)}")
    log(f"  Fichier sauvegardé      : {output_path}")
 
    if stats["valid_windows"] > 0:
        log(f"\n  Distribution glycemic_label :")
        for label, count in dataset["glycemic_label"].value_counts().items():
            pct = 100 * count / len(dataset)
            log(f"    {label:<15} : {count:>4} ({pct:.1f}%)")
 
        log(f"\n  Distribution par groupe :")
        group_col = "bio_group" if "bio_group" in dataset.columns else None
        if group_col:
            for grp, count in dataset[group_col].value_counts().items():
                log(f"    {grp:<15} : {count:>4} fenêtres")
 
    return dataset
 




# Téléchrgement du csv 
if __name__ == "__main__":
    dataset = build_meal_windows_dataset(
        data_raw_dir=CONFIG["data_raw_dir"],
        bio_path=CONFIG["bio_path"],
        output_dir=CONFIG["output_dir"],
        output_filename=CONFIG["output_filename"],
    )
    print(f"\nDataset prêt : {dataset.shape[0]} fenêtres × {dataset.shape[1]} colonnes")
  