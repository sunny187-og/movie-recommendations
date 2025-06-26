# proj.py

import pandas as pd
import ast
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Load CSVs ---
metadata = pd.read_csv('movies_metadata.csv', low_memory=False)
ratings = pd.read_csv('ratings_small.csv')
credits = pd.read_csv('credits.csv')
keywords = pd.read_csv('keywords.csv')

# --- Preprocessing ---
metadata = metadata[pd.to_numeric(metadata['id'], errors='coerce').notna()]

metadata['id'] = metadata['id'].astype(int)
credits['id'] = credits['id'].astype(int)
keywords['id'] = keywords['id'].astype(int)

movies = metadata.merge(credits, on='id').merge(keywords, on='id').head(5000)

def extract_names(json_str, key='name', topn=None):
    try:
        items = ast.literal_eval(json_str)
        names = [item[key] for item in items]
        return ' '.join(names[:topn]) if topn else ' '.join(names)
    except:
        return ''

def get_director(crew_str):
    try:
        crew = ast.literal_eval(crew_str)
        for member in crew:
            if member['job'] == 'Director':
                return member['name']
        return ''
    except:
        return ''

movies['genres'] = movies['genres'].apply(extract_names)
movies['keywords'] = movies['keywords'].apply(extract_names)
movies['top_actors'] = movies['cast'].apply(lambda x: extract_names(x, topn=3))
movies['director'] = movies['crew'].apply(get_director)

# --- TF-IDF Vectorization ---
def create_soup(row):
    return f"{row['genres']} {row['keywords']} {row['top_actors']} {row['director']}"

movies['soup'] = movies.apply(create_soup, axis=1)
tfidf = TfidfVectorizer(stop_words='english')
soup_matrix = tfidf.fit_transform(movies['soup'].fillna(''))

movies = movies[['id', 'title', 'soup']].rename(columns={'id': 'movieId'})
movies.reset_index(drop=True, inplace=True)

# --- Content-Based Filtering ---
def get_content_recs(title, topn=10):
    idx = movies[movies['title'].str.lower() == title.lower()].index
    if len(idx) == 0:
        return pd.DataFrame(columns=['movieId', 'title'])
    idx = idx[0]
    sim_scores = list(enumerate(cosine_similarity(soup_matrix[idx], soup_matrix).flatten()))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:topn+1]
    return movies.iloc[[i[0] for i in sim_scores]][['movieId', 'title']]

# --- Cold Start + Mood Recommendation (Content-based only) ---
mood_genre_map = {
    'happy': ['Comedy', 'Family', 'Adventure'],
    'sad': ['Drama', 'Romance'],
    'angry': ['Action', 'Thriller'],
    'bored': ['Fantasy', 'Sci-Fi', 'Mystery'],
    'romantic': ['Romance', 'Drama'],
    'excited': ['Action', 'Adventure'],
    'scared': ['Horror', 'Thriller'],
    'inspired': ['Biography', 'History', 'Documentary']
}

def cold_start_hybrid_with_mood(fav_genres=None, fav_movie=None, fav_actor=None, fav_director=None, mood=None, topn=10):
    preference_text = ''
    if fav_genres: preference_text += ' '.join(fav_genres) + ' '
    if fav_movie:
        row = movies[movies['title'].str.lower() == fav_movie.lower()]
        if not row.empty:
            preference_text += row.iloc[0]['soup'] + ' '
    if fav_actor: preference_text += fav_actor + ' '
    if fav_director: preference_text += fav_director + ' '
    if mood: preference_text += ' '.join(mood_genre_map.get(mood.lower(), [])) + ' '

    if not preference_text.strip():
        return pd.DataFrame(columns=['movieId', 'title'])

    profile_vector = tfidf.transform([preference_text])
    content_sim = cosine_similarity(profile_vector, soup_matrix).flatten()
    top_indices = content_sim.argsort()[-topn:][::-1]
    return movies.iloc[top_indices][['movieId', 'title']]
