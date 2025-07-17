from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np
import pickle

# Import ML utils from recommender_utils.py
from recommender_utils import (
    get_closest_product_name,
    get_recommendations,
    hybrid_recommendation,
    recommend_for_user,
    recommend_from_cluster,
    recommend_similar_products
)

app = Flask(__name__)

# Load data and models
df = pd.read_csv('cleaned_data_for_model/clean_data_4090.csv')
trending_df = pd.read_csv('cleaned_data_for_model/trending_products.csv')
cosine_sim = np.load('models/cosine_sim.npy')

tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
svd_model = pickle.load(open('models/svd_model.pkl', 'rb'))
random_forest_model = joblib.load('models/random_forest_model.pkl')
nearest_neighbors = joblib.load('models/nearest_neighbors_model.pkl')
kmeans_model = joblib.load('models/kmeans_model.pkl')

# Home → Smart Product Search
@app.route('/')
def index():
    return render_template('index.html')

# ⭐ Rating Predictor
@app.route('/rating', methods=['GET', 'POST'])
def rating():
    predicted_rating = None
    if request.method == 'POST':
        # For demo: You could use features and call random_forest_model.predict(X)
        features = {}  # To be defined from request.form
        # predicted_rating = predict_rating(features)
    return render_template('rating.html', prediction=predicted_rating)

# 🔁 Similar Products (KMeans or NearestNeighbors)
@app.route('/similar', methods=['GET', 'POST'])
def similar():
    results = []
    if request.method == 'POST':
        name = request.form['product_name']
        closest = get_closest_product_name(name, df)
        if closest:
            results = recommend_similar_products(closest, df, top_n=5)
    return render_template('similar.html', products=results)

# 🔀 Hybrid Recommendation (TF-IDF + SVD)
@app.route('/recommend', methods=['POST'])
def recommend():
    product_name = request.form['product_name']

    # Step 1: Fuzzy match the product name
    closest_name = get_closest_product_name(product_name, df)

    # Step 2: Run content-based recommender
    if closest_name:
        recommendations_df = get_recommendations(closest_name, df, cosine_sim, top_n=10, threshold=0.3)
        
        # Step 3: If result is a string (error), show error
        if isinstance(recommendations_df, str):
            return render_template('index.html', error=recommendations_df)

        # Step 4: Convert to list of dicts for Jinja
        recommendations = recommendations_df.to_dict(orient='records')

        # ✅ No need to split 'ImageURL' anymore
        return render_template('index.html',
                               recommendations=recommendations,
                               input_name=product_name,
                               closest_name=closest_name)
    else:
        return render_template('index.html', error=f"No match found for '{product_name}'")




# 📈 Trending Products Page
@app.route('/trending')
def trending():
    products = trending_df.copy()

    # Convert to dicts
    product_list = products.to_dict(orient='records')

    return render_template('trending.html', products=product_list)


if __name__ == '__main__':
    app.run(debug=True)
