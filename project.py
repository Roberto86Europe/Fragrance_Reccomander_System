import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Caricamento del catalogo dal file CSV
file_path = "C:\VsCode\myenv\sistema_di_raccomandazione\catalogo_profumi.csv"  # Modifica il percorso
catalogo = pd.read_csv(file_path)

# Pulizia dei nomi delle colonne
catalogo.columns = catalogo.columns.str.strip()

# Aggiunta di una colonna 'genere'
def assegna_genere(row):
    if 200 <= row['profumo'] < 400:
        return 'uomo'
    elif row['profumo'] < 200 or row['profumo'] >= 400:
        return 'donna'
    return 'unisex'

catalogo['genere'] = catalogo.apply(assegna_genere, axis=1)

# Pre-elaborazione: Creazione di una descrizione unificata
catalogo['descrizione'] = (
    catalogo['famiglia_olfattiva'] + " " +
    catalogo['note_di_testa'].fillna('') + " " +
    catalogo['note_di_cuore'].fillna('') + " " +
    catalogo['note_di_fondo'].fillna('')
)

# Trasformazione delle descrizioni in vettori TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(catalogo['descrizione'])

# Calcolo della matrice di similarità
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Funzione per ottenere raccomandazioni
def raccomanda_profumi(profumo_id, top_n=2):
    try:
        profumo_idx = catalogo[catalogo['profumo'] == profumo_id].index[0]
        profumo_genere = catalogo.iloc[profumo_idx]['genere']
    except IndexError:
        return f"Il profumo con ID {profumo_id} non è stato trovato nel catalogo."
    
    similarity_scores = list(enumerate(similarity_matrix[profumo_idx]))
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    recommended_indices = [i[0] for i in sorted_scores[1:] if catalogo.iloc[i[0]]['genere'] == profumo_genere][:top_n]
    recommended_ids = catalogo.iloc[recommended_indices][['profumo', 'famiglia_olfattiva']].values.tolist()
    return recommended_ids

# Interfaccia Streamlit
st.title("Sistema di Raccomandazione Profumi")
st.write("""
Questo sistema ti consente di ottenere raccomandazioni personalizzate 
sulla base di un profumo esistente.
""")

# Input utente
profumo_id = st.number_input("Inserisci l'ID del profumo:", min_value=1, step=1)

# Bottone per ottenere raccomandazioni
if st.button("Ottieni_Raccomandazioni"):
    raccomandazioni = raccomanda_profumi(profumo_id)
    if isinstance(raccomandazioni, str):
        st.error(raccomandazioni)
    else:
        st.success("Ecco i profumi consigliati:")
        for rec in raccomandazioni:
            st.write(f"**ID:** {rec[0]}, **Famiglia Olfattiva:** {rec[1]}")
