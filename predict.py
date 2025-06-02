import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
import os

# 🔹 1️⃣ Charger le fichier CSV (assurez-vous qu'il est dans le bon dossier)
file_path = "diseases.csv"  # Remplace par le chemin exact
if not os.path.exists(file_path):
    print(f"Erreur : Le fichier '{file_path}' est introuvable.")
    exit()

df = pd.read_csv(file_path)

# 🔹 2️⃣ Extraire les colonnes (maladies et symptômes)
symptom_columns = df.columns[1:]  # Exclure la colonne "Disease"
X = df[symptom_columns].values  # Matrice des symptômes (0 ou 1)
y = df["Disease"].values  # Liste des maladies

# 🔹 3️⃣ Créer le modèle de régression logistique multinomiale
model = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=500)
model.fit(X, y)

# 🔹 4️⃣ Afficher les symptômes avec un code pour l'utilisateur
print("\n📌 Liste des symptômes disponibles :")
symptom_dict = {i+1: symptom for i, symptom in enumerate(symptom_columns)}
for code, symptom in symptom_dict.items():
    print(f"{code}: {symptom}")

# 🔹 5️⃣ Demander à l'utilisateur d'entrer les symptômes observés
try:
    selected_codes = input("\nEntrez les codes des symptômes séparés par des espaces : ")
    selected_symptoms = [int(code) for code in selected_codes.split()]
except ValueError:
    print("Erreur : Veuillez entrer uniquement des nombres valides.")
    exit()

# 🔹 6️⃣ Construire le tableau des symptômes du patient
patient_symptoms = np.zeros(len(symptom_columns))  # Liste remplie de 0

for code in selected_symptoms:
    if code in symptom_dict:
        symptom_name = symptom_dict[code]
        index = list(symptom_columns).index(symptom_name)
        patient_symptoms[index] = 1  # Mettre 1 pour les symptômes sélectionnés

# 🔹 7️⃣ Prédire les maladies avec leurs probabilités
probs = model.predict_proba([patient_symptoms])[0]
disease_probabilities = dict(zip(model.classes_, probs))

# 🔹 8️⃣ Trier et afficher les résultats
sorted_diseases = sorted(disease_probabilities.items(), key=lambda x: x[1], reverse=True)

print("\n🔍 Résultats de la prédiction :")
for disease, probability in sorted_diseases:
    print(f"{disease}: {probability * 100:.2f}%")
