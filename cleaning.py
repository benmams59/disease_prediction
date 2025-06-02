import pandas as pd

# Charger le fichier CSV
df = pd.read_csv("DiseaseAndSymptoms.csv")

# Obtenir toutes les valeurs uniques des symptômes
symptoms = pd.unique(df.iloc[:, 1:].values.ravel())
symptoms = [symptom for symptom in symptoms if pd.notna(symptom)]  # Supprimer les valeurs NaN

# Créer un nouveau DataFrame avec les mêmes lignes et ajouter les colonnes de symptômes
df_transformed = df.copy()
for symptom in symptoms:
    df_transformed[symptom] = df.iloc[:, 1:].apply(lambda x: int(symptom in x.values), axis=1)

# Supprimer les anciennes colonnes de symptômes
df_transformed.drop(columns=df.columns[1:], inplace=True)

# Sauvegarder le fichier transformé
df_transformed.to_csv("diseases.csv", index=False)

print("Transformation terminée. Le fichier 'diseases.csv' est généré.")