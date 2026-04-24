#Ce code me permet d'avoir une catégorie directement ajouté dans le fichier bio.csv classifiant les 45 patients selon les 3 types : sujets sains, prédiabétiques ou diabétiques de type 2.
# Pour faire cette classification, je m'aide de la colonne "A1c PDL (Lab)" qui est déjà présente dans bio.csv. Elle décrit une fourchette des valeurs d'HbA1c mesurées au début de l'étude en 
# mmol/mol (hémoglobine glyquée) 
# La classification se fait selon les pourcentages suivants : 
# Healthy si HbA1c < 5.7%
# Prediabetes si 5.7% ≤ HbA1c ≤ 6.4%
# Type 2 Diabetes (T2D) si HbA1c > 6.4%.

import pandas as pd

# Lire le fichier bio.csv
bio = pd.read_csv("data/raw/cgmacros/bio.csv")

def get_group(hba1c):
    if hba1c < 5.7:
        return "healthy"
    elif hba1c <= 6.4:
        return "prediabetes"
    else:
        return "t2d"

#Création de la nouvelle colonne "group" en appliquant la fonction get_group à la colonne "A1c PDL (Lab)"
bio["group"] = bio["A1c PDL (Lab)"].apply(get_group)

# Déplacer la colonne en 2e position
cols = bio.columns.tolist()
cols.insert(1, cols.pop(cols.index("group")))
bio = bio[cols]

# Sauvegarder le nouveau fichier
bio.to_csv("data/processed/bio_with_group.csv", index=False)

print("Fichier créé : data/processed/bio_with_group.csv")
print(bio.head())

