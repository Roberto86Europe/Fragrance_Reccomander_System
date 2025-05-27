from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the perfume catalog
catalog = pd.read_csv("catalogo_profumi.csv")

# Clean column names
catalog.columns = catalog.columns.str.strip()

# Preprocessing: Create a unified description
catalog['description'] = (
    catalog['famiglia_olfattiva'].fillna('') + " " +
    catalog['note_di_testa'].fillna('') + " " +
    catalog['note_di_cuore'].fillna('') + " " +
    catalog['note_di_fondo'].fillna('')
)

# Convert descriptions to TF-IDF vectors
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(catalog['description'])

# Compute similarity matrix
similarity_matrix = cosine_similarity(tfidf_matrix)

# API Endpoint
@app.route("/recommend", methods=["GET"])
def recommend():
    try:
        perfume_id = int(request.args.get("id"))
        idx = catalog[catalog['profumo'] == perfume_id].index[0]
        scores = list(enumerate(similarity_matrix[idx]))
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:4]
        recommendations = catalog.iloc[[i[0] for i in sorted_scores]]['profumo'].tolist()
        return jsonify(recommendations)
    except:
        return jsonify({"error": "Invalid or not found ID."})

if __name__ == "__main__":
    app.run()
