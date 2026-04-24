"""
column_description_meal_window_builder.py
=====================
Génère un CSV décrivant colonne par colonne le fichier
meal_windows_dataset.csv produit par meal_window_builder.py.

Usage : python column_description_meal_window_builder.py
Sortie : column_description_meal_windows.csv

Auteur : Palliere Raphael — E4 Bio
"""

import pandas as pd

COLUMNS = [


    #BLOC 1 — Identifiants et contexte du repas
    {
        "Colonne":        "patient_id",
        "Bloc":           "Identifiant",
        "Type":           "Entier",
        "Description":    "Numéro unique du patient (1 à 49, hors 24/25/37/40). Utilisé pour le split train/test : toutes les fenêtres d'un même patient restent dans le même fold.",
        "Utilisation ML": "Clé de groupement — ne jamais utiliser comme feature",
    },
    {
        "Colonne":        "meal_time",
        "Bloc":           "Identifiant",
        "Type":           "Datetime",
        "Description":    "Horodatage du début du repas (t=0). Sert de référence pour construire toute la fenêtre temporelle. Les dates sont décalées artificiellement (anonymisation CGMacros).",
        "Utilisation ML": "Référence temporelle — ne pas utiliser brut comme feature",
    },
    {
        "Colonne":        "meal_type",
        "Bloc":           "Identifiant",
        "Type":           "Catégoriel (string)",
        "Description":    "Type de repas : Breakfast / Lunch / Dinner / Snacks. Encode un contexte circadien important : la réponse glycémique au même repas peut différer selon l'heure de la journée.",
        "Utilisation ML": "Feature catégorielle — encodage one-hot recommandé",
    },

    #BLOC 2 — encodage temporel cyclique
    {
        "Colonne":        "hour_sin",
        "Bloc":           "Encodage temporel",
        "Type":           "Float",
        "Description":    "Partie sinus de l'heure du repas encodée cycliquement sur 24h. Formule : sin(2π × heure / 24). Permet au modèle de comprendre que 23h et 0h sont proches.",
        "Utilisation ML": "Feature numérique — input modèle",
    },
    {
        "Colonne":        "hour_cos",
        "Bloc":           "Encodage temporel",
        "Type":           "Float",
        "Description":    "Partie cosinus de l'heure du repas encodée cycliquement sur 24h. Formule : cos(2π × heure / 24). Complémentaire à hour_sin.",
        "Utilisation ML": "Feature numérique — input modèle",
    },
    {
        "Colonne":        "dow_sin",
        "Bloc":           "Encodage temporel",
        "Type":           "Float",
        "Description":    "Partie sinus du jour de la semaine encodé cycliquement sur 7 jours. Formule : sin(2π × jour / 7). 0=lundi, 6=dimanche.",
        "Utilisation ML": "Feature numérique — input modèle",
    },
    {
        "Colonne":        "dow_cos",
        "Bloc":           "Encodage temporel",
        "Type":           "Float",
        "Description":    "Partie cosinus du jour de la semaine encodé cycliquement sur 7 jours. Formule : cos(2π × jour / 7). Complémentaire à dow_sin.",
        "Utilisation ML": "Feature numérique — input modèle",
    },

    
    #BLOC 3 — descripteurs nutritionnels du repas
    {
        "Colonne":        "carbs",
        "Bloc":           "Nutrition",
        "Type":           "Float (grammes)",
        "Description":    "Quantité de glucides du repas en grammes. Macronutriment le plus directement lié à la réponse glycémique : les glucides sont convertis en glucose et absorbés dans le sang. Feature nutritionnelle la plus prédictive de la PPGR.",
        "Utilisation ML": "Feature numérique prioritaire — input modèle",
    },
    {
        "Colonne":        "protein",
        "Bloc":           "Nutrition",
        "Type":           "Float (grammes)",
        "Description":    "Quantité de protéines du repas en grammes. Effet modérateur non linéaire sur la glycémie : stimule à la fois l'insuline et le glucagon. L'effet net dépend du contexte métabolique.",
        "Utilisation ML": "Feature numérique — input modèle",
    },
    {
        "Colonne":        "fat",
        "Bloc":           "Nutrition",
        "Type":           "Float (grammes)",
        "Description":    "Quantité de lipides du repas en grammes. Ralentit la vidange gastrique : décale et aplatit le pic glycémique (pic plus tardif et moins élevé).",
        "Utilisation ML": "Feature numérique — input modèle",
    },
    {
        "Colonne":        "fiber",
        "Bloc":           "Nutrition",
        "Type":           "Float (grammes)",
        "Description":    "Quantité de fibres alimentaires du repas en grammes. Ralentit l'absorption des glucides et réduit l'amplitude du pic glycémique postprandial. Quelques NaN présents.",
        "Utilisation ML": "Feature numérique — input modèle (imputation si NaN)",
    },
    
    {
        "Colonne":        "calories",
        "Bloc":           "Nutrition",
        "Type":           "Float (kcal)",
        "Description":    "Apport calorique total du repas. Partiellement redondant avec carbs/protein/fat (glucides×4 + protéines×4 + lipides×9). À évaluer pour colinéarité.",
        "Utilisation ML": "Feature numérique — évaluer la colinéarité avec les macros",
    },

     
    #BLOC 4 — caractéristiques CGM pré-repas agrégées
    {
        "Colonne":        "cgm_at_meal",
        "Bloc":           "CGM agrégé",
        "Type":           "Float (mg/dL)",
        "Description":    "Glycémie Abbott FreeStyle Libre au moment exact du repas (t=0). Valeur de départ de la réponse glycémique — fortement prédictive de la valeur absolue à t+60 min.",
        "Utilisation ML": "Feature numérique prioritaire — input modèle",
    },
    {
        "Colonne":        "cgm_pre_mean",
        "Bloc":           "CGM agrégé",
        "Type":           "Float (mg/dL)",
        "Description":    "Moyenne de la glycémie sur les 60 minutes précédant le repas. Résume le niveau glycémique basal du patient avant le repas.",
        "Utilisation ML": "Feature numérique — input baseline tabulaire",
    },
    {
        "Colonne":        "cgm_pre_std",
        "Bloc":           "CGM agrégé",
        "Type":           "Float (mg/dL)",
        "Description":    "Écart-type de la glycémie sur les 60 minutes précédant le repas. Mesure la variabilité glycémique pré-repas. Un écart-type élevé peut indiquer une instabilité métabolique.",
        "Utilisation ML": "Feature numérique — input baseline tabulaire",
    },
    {
        "Colonne":        "cgm_pre_min",
        "Bloc":           "CGM agrégé",
        "Type":           "Float (mg/dL)",
        "Description":    "Valeur minimale de la glycémie sur les 60 minutes précédant le repas. Permet de détecter un épisode hypoglycémique pré-repas.",
        "Utilisation ML": "Feature numérique — input baseline tabulaire",
    },
    {
        "Colonne":        "cgm_pre_max",
        "Bloc":           "CGM agrégé",
        "Type":           "Float (mg/dL)",
        "Description":    "Valeur maximale de la glycémie sur les 60 minutes précédant le repas.",
        "Utilisation ML": "Feature numérique — input baseline tabulaire",
    },
    {
        "Colonne":        "cgm_slope_15",
        "Bloc":           "CGM agrégé",
        "Type":           "Float (mg/dL/min)",
        "Description":    "Pente linéaire de la glycémie sur les 15 dernières minutes avant le repas. Positive = glycémie en hausse, négative = en baisse, ~0 = stable. Capture la tendance immédiate.",
        "Utilisation ML": "Feature numérique — input baseline et modèle séquentiel",
    },
    {
        "Colonne":        "cgm_slope_30",
        "Bloc":           "CGM agrégé",
        "Type":           "Float (mg/dL/min)",
        "Description":    "Pente linéaire de la glycémie sur les 30 dernières minutes avant le repas. Tendance glycémique à plus long terme que cgm_slope_15.",
        "Utilisation ML": "Feature numérique — input baseline et modèle séquentiel",
    },
    {
        "Colonne":        "n_valid_pre",
        "Bloc":           "CGM agrégé",
        "Type":           "Entier",
        "Description":    "Nombre de points CGM valides (non-NaN) dans la fenêtre pré-repas de 60 min. Indicateur de qualité du signal. Minimum requis : 30 points.",
        "Utilisation ML": "Indicateur qualité — ne pas utiliser comme feature prédictive",
    },

    #BLOC 5 — Biomarqueurs patient
    {
        "Colonne":        "bio_group",
        "Bloc":           "Biomarqueurs patient",
        "Type":           "Catégoriel (string)",
        "Description":    "Groupe métabolique du patient : healthy / prediabetes / t2d. Déterminé à partir du taux d'HbA1c mesuré au jour 1 de l'étude.",
        "Utilisation ML": "Feature catégorielle — encodage one-hot ou ordinal",
    },
    {
        "Colonne":        "bio_group_encoded",
        "Bloc":           "Biomarqueurs patient",
        "Type":           "Entier (0/1/2)",
        "Description":    "Groupe métabolique encodé ordinalement : 0=healthy, 1=prediabetes, 2=t2d. L'ordre est cliniquement justifié (progression de la maladie).",
        "Utilisation ML": "Feature numérique ordinale — input modèle",
    },
    {
        "Colonne":        "bio_gender_encoded",
        "Bloc":           "Biomarqueurs patient",
        "Type":           "Entier (0/1)",
        "Description":    "Genre encodé : 0=Masculin, 1=Féminin. Influence la répartition des masses tissulaires et la cinétique métabolique basale.",
        "Utilisation ML": "Feature numérique binaire — input modèle",
    },
    {
        "Colonne":        "bio_Age",
        "Bloc":           "Biomarqueurs patient",
        "Type":           "Entier (années)",
        "Description":    "Âge du patient en années. La sensibilité à l'insuline diminue avec l'âge.",
        "Utilisation ML": "Feature numérique — input modèle",
    },
    {
        "Colonne":        "bio_BMI",
        "Bloc":           "Biomarqueurs patient",
        "Type":           "Float (kg/m²)",
        "Description":    "Indice de Masse Corporelle. Fortement corrélé à l'insulinorésistance. Remplace les variables poids et taille bruts (redondants).",
        "Utilisation ML": "Feature numérique — input modèle",
    },
    {
        "Colonne":        "bio_A1c PDL (Lab)",
        "Bloc":           "Biomarqueurs patient",
        "Type":           "Float (%)",
        "Description":    "HbA1c en %, mesurée au jour 1. Reflète le contrôle glycémique moyen sur 2-3 mois. Biomarqueur le plus discriminant entre les 3 groupes. Seuils : <5.7%=healthy, 5.7-6.4%=prediabetes, >6.4%=t2d.",
        "Utilisation ML": "Feature numérique prioritaire — input modèle",
    },
    {
        "Colonne":        "bio_Fasting GLU - PDL (Lab)",
        "Bloc":           "Biomarqueurs patient",
        "Type":           "Float (mg/dL)",
        "Description":    "Glycémie à jeun mesurée en laboratoire au jour 1. Évalue la capacité basale de l'organisme à réguler sa glycémie après une nuit sans apport calorique.",
        "Utilisation ML": "Feature numérique — input modèle",
    },
    {
        "Colonne":        "bio_Insulin",
        "Bloc":           "Biomarqueurs patient",
        "Type":           "Float (µU/mL)",
        "Description":    "Taux d'insuline à jeun mesuré au jour 1. Un taux élevé signale une hyperinsulinémie compensatoire, marqueur précoce de résistance à l'insuline.",
        "Utilisation ML": "Feature numérique — input modèle",
    },
    {
        "Colonne":        "bio_Triglycerides",
        "Bloc":           "Biomarqueurs patient",
        "Type":           "Float (mg/dL)",
        "Description":    "Triglycérides sanguins mesurés au jour 1. Marqueur du syndrome métabolique. Retenu car peu colinéaire avec les autres biomarqueurs conservés.",
        "Utilisation ML": "Feature numérique — input modèle",
    },
    {
        "Colonne":        "bio_HDL",
        "Bloc":           "Biomarqueurs patient",
        "Type":           "Float (mg/dL)",
        "Description":    "Cholestérol HDL mesuré au jour 1. Corrélé négativement à l'insulinorésistance : un HDL bas est un signal de risque métabolique élevé.",
        "Utilisation ML": "Feature numérique — input modèle",
    },


    #BLOC 6 — SÉQUENCE CGM PRÉ-REPAS (input LSTM/GRU)
    {
        "Colonne":        "cgm_t-60 à cgm_t-1",
        "Bloc":           "Séquence CGM pré-repas",
        "Type":           "60 colonnes Float (mg/dL)",
        "Description":    "Séquence de 60 valeurs CGM Abbott, une par minute, de t-60min à t-1min avant le repas. cgm_t-60 = glycémie 60 min avant le repas (la plus ancienne). cgm_t-1 = glycémie 1 min avant le repas (la plus récente). Input temporel direct du modèle LSTM/GRU. Peut contenir des NaN si le signal CGM était absent.",
        "Utilisation ML": "Séquence input LSTM/GRU — normaliser par patient avant entraînement",
    },

    
    #BLOC 7 — SÉQUENCE CGM POST-REPAS (exploration future)
    {
        "Colonne":        "cgm_post_t1 à cgm_post_t90",
        "Bloc":           "Séquence CGM post-repas",
        "Type":           "90 colonnes Float (mg/dL)",
        "Description":    "Séquence de 90 valeurs CGM Abbott, une par minute, de t+1min à t+90min après le repas. Représente la courbe complète de la PPGR. Extrait pour exploration future (modèle seq-to-seq). Ne pas inclure dans les features d'entrée.",
        "Utilisation ML": "Réservé exploration future — ne pas inclure dans les features d'entrée",
    },


    #BLOC 8 — VARIABLES CIBLES
    {
        "Colonne":        "cgm_target_t30",
        "Bloc":           "Cible",
        "Type":           "Float (mg/dL)",
        "Description":    "Glycémie Abbott à t+30 min après le début du repas. Horizon court — correspond au début de la montée glycémique pour la plupart des repas mixtes.",
        "Utilisation ML": "Variable cible horizon court",
    },
    {
        "Colonne":        "cgm_target_t60",
        "Bloc":           "Cible",
        "Type":           "Float (mg/dL)",
        "Description":    "Glycémie Abbott à t+60 min après le début du repas. Horizon principal du projet — correspond approximativement au pic glycémique postprandial pour un repas standard. Cible de référence pour l'entraînement et l'évaluation.",
        "Utilisation ML": "Variable cible principale — horizon de référence",
    },
    {
        "Colonne":        "cgm_target_t90",
        "Bloc":           "Cible",
        "Type":           "Float (mg/dL)",
        "Description":    "Glycémie Abbott à t+90 min après le début du repas. Horizon long — début de la phase de retour à la normale. Utile pour évaluer la durée de l'hyperglycémie postprandiale.",
        "Utilisation ML": "Variable cible horizon long",
    },
    {
        "Colonne":        "glycemic_label",
        "Bloc":           "Cible",
        "Type":           "Catégoriel (string)",
        "Description":    "Label clinique dérivé de cgm_target_t60. 4 classes : hypo (<70 mg/dL), normal (70-140 mg/dL), hyper_mild (140-180 mg/dL), hyper_severe (>180 mg/dL). Utilisé pour la tâche de classification secondaire. Classes naturellement déséquilibrées (normal surreprésenté).",
        "Utilisation ML": "Variable cible classification — gérer le déséquilibre (SMOTE ou class_weight)",
    },
]


if __name__ == "__main__":
    df = pd.DataFrame(COLUMNS)
    output_path = "data/processed/column_description_meal_window.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Fichier généré : {output_path}")
    print(f"{len(df)} colonnes décrites | blocs : {df['Bloc'].unique().tolist()}")