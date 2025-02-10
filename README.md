# Fragrance Recommender System

This project implements a **Fragrance Recommendation System** using collaborative filtering and content-based techniques. The system is designed to recommend similar perfumes based on the user's input, taking into account the perfume's olfactory family, notes, and gender.

## Features
- **Personalized Recommendations**: Based on the input perfume ID, the system suggests similar fragrances.
- **Gender-specific Filtering**: Ensures recommendations are gender-appropriate (men, women, unisex).
- **Interactive User Interface**: Built with Streamlit for a simple and intuitive experience.

## How It Works
1. **Data Preprocessing**: The catalog of perfumes is loaded from a CSV file (`catalogo_profumi.csv`), and a unified description is created based on the perfume's olfactory family and notes.
2. **Similarity Matrix**: TF-IDF is used to vectorize descriptions, and cosine similarity is computed between all perfumes.
3. **Recommendations**: Based on a given perfume ID, the system ranks similar perfumes and filters results by gender.

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/Fragrance_Recommender_System.git

2. Navigate to the project directory
   
  cd Fragrance_Recommender_System

3. Install required dependencies

   pip install -r requirements.txt

4. Run the application

   streamlit run fragrance_recommender.py


File Structure
catalogo_profumi.csv: Contains the perfume catalog with details like olfactory family, notes, gender, and sales data.
fragrance_recommender.py: Main Python script implementing the recommendation logic and Streamlit interface.

Example Output
Input: Perfume ID = 10
Recommended Perfumes:
- ID: 15, Olfactory Family: Floral
- ID: 22, Olfactory Family: Woody

Requirements
Python 3.x
Libraries: pandas, scikit-learn, streamlit



Author
Developed by Roberto Bombardieri.For questions or feedback, feel free to contact me.



   
