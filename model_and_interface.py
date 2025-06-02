import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Fonction pour charger et entraîner le modèle
def train_model(file_path):
    df = pd.read_csv(file_path)
    symptom_columns = df.columns[1:]  # Exclure "Disease"
    X = df[symptom_columns].values
    y = df["Disease"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=500)
    model.fit(X_train, y_train)
    
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, symptom_columns, accuracy, df

# Interface utilisateur avec Streamlit
st.title("Prédiction de Maladies via Symptômes")

# Charger les données
file_path = "diseases.csv"
model, symptom_columns, accuracy, df = train_model(file_path)

st.sidebar.header("Sélectionnez vos symptômes")
selected_symptoms = st.sidebar.multiselect("Choisissez les symptômes que vous ressentez", symptom_columns)

if st.sidebar.button("Prédire la Maladie"):
    patient_symptoms = np.zeros(len(symptom_columns))
    for symptom in selected_symptoms:
        index = list(symptom_columns).index(symptom)
        patient_symptoms[index] = 1
    
    probs = model.predict_proba([patient_symptoms])[0]
    disease_probabilities = dict(zip(model.classes_, probs))
    sorted_diseases = sorted(disease_probabilities.items(), key=lambda x: x[1], reverse=True)
    
    st.subheader("Résultats de la Prédiction")
    for disease, probability in sorted_diseases:
        st.write(f"**{disease}**: {probability * 100:.2f}%")

st.sidebar.write(f"**Précision du modèle :** {accuracy * 100:.2f}%")

# Section cachée pour afficher les maladies et leurs symptômes
with st.expander("Voir les maladies et leurs symptômes"):
    selected_disease = st.selectbox("Sélectionnez une maladie", df["Disease"].unique())
    if selected_disease:
        symptoms_for_disease = df[df["Disease"] == selected_disease].iloc[:, 1:].sum()
        symptoms_present = symptoms_for_disease[symptoms_for_disease > 0].index.tolist()
        st.write(f"**Symptômes associés à {selected_disease}:**")
        st.write(", ".join(symptoms_present))


