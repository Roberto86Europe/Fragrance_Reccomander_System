import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load the catalog from the CSV file
file_path = "file.csv"  # Modify the path
catalog = pd.read_csv(file_path)

# Clean column names
catalog.columns = catalog.columns.str.strip()

# Add a 'gender' column
def assign_gender(row):
    if 200 <= row['perfume'] < 400:
        return 'men'
    elif row['perfume'] < 200 or row['perfume'] >= 400:
        return 'women'
    return 'unisex'

catalog['gender'] = catalog.apply(assign_gender, axis=1)

# Preprocessing: Create a unified description
catalog['description'] = (
    catalog['olfactory_family'] + " " +
    catalog['top_notes'].fillna('') + " " +
    catalog['heart_notes'].fillna('') + " " +
    catalog['base_notes'].fillna('')
)

# Transform descriptions into TF-IDF vectors
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(catalog['description'])

# Compute similarity matrix
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get recommendations
def recommend_perfumes(perfume_id, top_n=2):
    try:
        perfume_idx = catalog[catalog['perfume'] == perfume_id].index[0]
        perfume_gender = catalog.iloc[perfume_idx]['gender']
    except IndexError:
        return f"The perfume with ID {perfume_id} was not found in the catalog."
    
    similarity_scores = list(enumerate(similarity_matrix[perfume_idx]))
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    recommended_indices = [i[0] for i in sorted_scores[1:] if catalog.iloc[i[0]]['gender'] == perfume_gender][:top_n]
    recommended_ids = catalog.iloc[recommended_indices][['perfume', 'olfactory_family']].values.tolist()
    return recommended_ids

