import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configuration de la page Streamlit
st.set_page_config(page_title="Recommandation de Parfums", layout="wide")


# Charger les données
@st.cache_data
def load_data():
    return pd.read_excel("Classement_Parfums_91.xlsx")


df = load_data()

# Fusionner les caractéristiques textuelles pour le modèle TF-IDF
df["features"] = df["Personnalité"] + " " + df["Occasion"] + " " + df["Notes dominantes"] + " " + df["Intensité"]

# Initialiser le vectorizer TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["features"])


# Fonction pour recommander les 3 meilleurs parfums
def recommander_parfum(personnalite, occasion, notes, intensite, poids):
    input_text = (f"{personnalite} " * poids[0] +
                  f"{occasion} " * poids[1] +
                  f"{notes} " * poids[2] +
                  f"{intensite} " * poids[3])

    input_vector = vectorizer.transform([input_text])
    scores = cosine_similarity(input_vector, tfidf_matrix)[0]

    # Trouver les 3 meilleurs indices
    top_3_indices = scores.argsort()[-3:][::-1]
    top_3_parfums = df.iloc[top_3_indices]["Nom du parfum"].values
    top_3_scores = scores[top_3_indices]

    return list(zip(top_3_parfums, top_3_scores))


# Interface utilisateur divisée en deux colonnes
col1, col2 = st.columns(2)

# 🔍 **Recherche Simple**
with col1:
    st.header("🔍 Recherche Simple")

    # Questions identiques pour les deux recherches
    notes = st.selectbox("Quelle note olfactive préférez-vous ?", df["Notes dominantes"].unique(), key="simple_notes")
    personnalite = st.selectbox("Quel est votre type de personnalité ?", df["Personnalité"].unique(),
                                key="simple_perso")
    occasion = st.selectbox("Quelle est l'occasion ?", df["Occasion"].unique(), key="simple_occa")
    intensite = st.selectbox("Quelle intensité préférez-vous ?", df["Intensité"].unique(), key="simple_intens")

    resultats = df[
        (df["Notes dominantes"] == notes) &
        (df["Personnalité"] == personnalite) &
        (df["Occasion"] == occasion) &
        (df["Intensité"] == intensite)
        ]

    if st.button("🔍 Trouver mon parfum (simple)"):
        if not resultats.empty:
            st.subheader("✨ Nous vous recommandons ces parfums :")
            for parfum in resultats["Nom du parfum"].head(3):
                st.write(f"- {parfum}")
        else:
            st.warning("Aucun parfum ne correspond exactement à vos critères. Essayez d'ajuster vos choix !")

# 🤖 **Recherche Recommandée (IA)**
with col2:
    st.header("🤖 Recherche Recommandée (IA)")

    # Questions identiques mais avec pondération
    notes_ia = st.selectbox("Quelle note olfactive préférez-vous ?", df["Notes dominantes"].unique(), key="ia_notes")
    personnalite_ia = st.selectbox("Quel est votre type de personnalité ?", df["Personnalité"].unique(), key="ia_perso")
    occasion_ia = st.selectbox("Quelle est l'occasion ?", df["Occasion"].unique(), key="ia_occa")
    intensite_ia = st.selectbox("Quelle intensité préférez-vous ?", df["Intensité"].unique(), key="ia_intens")

    poids = [
        st.slider("Importance de la personnalité", 1, 3, 2),
        st.slider("Importance de l'occasion", 1, 3, 2),
        st.slider("Importance des notes dominantes", 1, 3, 2),
        st.slider("Importance de l'intensité", 1, 3, 2)
    ]

    if st.button("🤖 Trouver mon parfum (IA)"):
        recommandations = recommander_parfum(personnalite_ia, occasion_ia, notes_ia, intensite_ia, poids)
        if recommandations:
            st.subheader("✨ Top 3 parfums recommandés :")
            for parfum, score in recommandations:
                st.write(f"- **{parfum}** (Score : {score:.2f})")
        else:
            st.warning("Aucune recommandation disponible avec ces critères. Essayez d'ajuster les pondérations !")
