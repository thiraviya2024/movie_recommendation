
# 🎬 Movie Recommendation System (Fully Fixed Final Code)

import pandas as pd
import numpy as np
import ast

# -------------------------------
# Step 1: Load Dataset
# -------------------------------
movies = pd.read_csv(r"D:\movie-recommender\tmdb_5000_movies.csv")
credits = pd.read_csv(r"D:\movie-recommender\tmdb_5000_credits.csv")

# -------------------------------
# Step 2: Fix Column Name
# -------------------------------
credits.rename(columns={'movie_id': 'id'}, inplace=True)

# -------------------------------
# Step 3: Merge Datasets
# -------------------------------
movies = movies.merge(credits, on='id')

# ✅ FIX: Handle duplicate title column
movies = movies.rename(columns={'title_x': 'title'})

# -------------------------------
# Step 4: Select Important Columns
# -------------------------------
movies = movies[['id','title','overview','genres','keywords','cast','crew']]

# -------------------------------
# Step 5: Handle Missing Values
# -------------------------------
movies.dropna(inplace=True)

# -------------------------------
# Step 6: Convert JSON Columns
# -------------------------------
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

# -------------------------------
# Step 7: Process Cast (Top 3 Actors)
# -------------------------------
def convert_cast(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter != 3:
            L.append(i['name'])
            counter += 1
    return L

movies['cast'] = movies['cast'].apply(convert_cast)

# -------------------------------
# Step 8: Extract Director
# -------------------------------
def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L

movies['crew'] = movies['crew'].apply(fetch_director)

# -------------------------------
# Step 9: Remove Spaces
# -------------------------------
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])

# -------------------------------
# Step 10: Create Tags
# -------------------------------
movies['overview'] = movies['overview'].apply(lambda x: x.split())

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# -------------------------------
# Step 11: Convert Tags to String
# -------------------------------
movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))

# -------------------------------
# Step 12: New DataFrame
# -------------------------------
new_df = movies[['id','title','tags']]

# -------------------------------
# Step 13: Vectorization
# -------------------------------
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

# -------------------------------
# Step 14: Similarity Matrix
# -------------------------------
from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vectors)

# -------------------------------
# Step 15: Recommendation Function
# -------------------------------
def recommend(movie):
    try:
        # ✅ Improved search (case-insensitive)
        movie_index = new_df[new_df['title'].str.lower() == movie.lower()].index[0]
        distances = similarity[movie_index]

        movies_list = sorted(list(enumerate(distances)),
                             reverse=True,
                             key=lambda x: x[1])[1:6]

        print(f"\n🎬 Top recommendations for '{movie}':\n")

        for i in movies_list:
            print(new_df.iloc[i[0]].title)

    except:
        print("❌ Movie not found. Please check the name.")

# -------------------------------
# Step 16: Run Program
# -------------------------------
if __name__ == "__main__":
    print("🎬 Movie Recommendation System")
    movie_name = input("Enter a movie name: ")
    recommend(movie_name)

