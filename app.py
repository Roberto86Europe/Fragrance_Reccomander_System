from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Caricamento del catalogo
catalogo = pd.read_csv("catalogo_profumi.csv")

# Pulizia dei nomi delle colonne
catalogo.columns = catalogo.columns.str.strip()

# Pre-elaborazione: Creazione di una descrizione unificata
catalogo['descrizione'] = (
    catalogo['famiglia_olfattiva'].fillna('') + " " +
    catalogo['note_di_testa'].fillna('') + " " +
    catalogo['note_di_cuore'].fillna('') + " " +
    catalogo['note_di_fondo'].fillna('')
)

# Trasformazione delle descrizioni in vettori TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(catalogo['descrizione'])

# Calcolo della matrice di similarità
similarity_matrix = cosine_similarity(tfidf_matrix)

# Endpoint API
@app.route("/raccomanda", methods=["GET"])
def raccomanda():
    try:
        profumo_id = int(request.args.get("id"))
        idx = catalogo[catalogo['profumo'] == profumo_id].index[0]
        scores = list(enumerate(similarity_matrix[idx]))
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:4]
        raccomandazioni = catalogo.iloc[[i[0] for i in sorted_scores]]['profumo'].tolist()
        return jsonify({"raccomandazioni": raccomandazioni})
    except:
        return jsonify({"errore": "ID non valido o non trovato."})

if __name__ == "__main__":
    app.run()
