
import pandas as pd
import numpy as np
import ast

# -------------------------------
# Load Dataset (Compressed Files)
# -------------------------------
movies = pd.read_csv("tmdb_5000_movies.csv.gz", compression='gzip')
credits = pd.read_csv("tmdb_5000_credits.csv.gz", compression='gzip')

credits.rename(columns={'movie_id': 'id'}, inplace=True)

movies = movies.merge(credits, on='id')
movies = movies.rename(columns={'title_x': 'title'})

movies = movies[['id','title','overview','genres','keywords','cast','crew']]
movies.dropna(inplace=True)

# -------------------------------
# Preprocessing
# -------------------------------
def convert(text):
    return [i['name'] for i in ast.literal_eval(text)]

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

def convert_cast(text):
    return [i['name'] for i in ast.literal_eval(text)[:3]]

movies['cast'] = movies['cast'].apply(convert_cast)

def fetch_director(text):
    return [i['name'] for i in ast.literal_eval(text) if i['job'] == 'Director']

movies['crew'] = movies['crew'].apply(fetch_director)

for col in ['genres','keywords','cast','crew']:
    movies[col] = movies[col].apply(lambda x: [i.replace(" ","") for i in x])

movies['overview'] = movies['overview'].apply(lambda x: x.split())

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))

new_df = movies[['id','title','tags']]

# -------------------------------
# Vectorization
# -------------------------------
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

similarity = cosine_similarity(vectors)

# -------------------------------
# Recommendation Function
# -------------------------------
def recommend(movie):
    try:
        movie_index = new_df[new_df['title'].str.lower() == movie.lower()].index[0]
        distances = similarity[movie_index]

        movies_list = sorted(list(enumerate(distances)),
                             reverse=True,
                             key=lambda x: x[1])[1:6]

        print(f"\n🎬 Top recommendations for '{movie}':\n")

        for i in movies_list:
            print(new_df.iloc[i[0]].title)

    except:
        print("❌ Movie not found.")

# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    movie_name = input("Enter a movie name: ")
    recommend(movie_name)

