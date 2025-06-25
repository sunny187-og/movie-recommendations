# proj.py

import pandas as pd
import ast
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader

# --- Load CSVs ---
metadata = pd.read_csv('movies_metadata.csv', low_memory=False)
ratings = pd.read_csv('ratings_small.csv')
credits = pd.read_csv('credits.csv')
keywords = pd.read_csv('keywords.csv')

# --- Preprocessing ---
metadata = metadata[metadata['id'].apply(lambda x: x.isdigit())]
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

# --- Collaborative Filtering ---
reader = Reader(rating_scale=(0.5, 5.0))
ratings = ratings[ratings['movieId'].isin(movies['movieId'])]
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()

model = SVD(n_factors=100, n_epochs=20)
model.fit(trainset)

def get_collab_recs(user_id, topn=10):
    watched = set(ratings[ratings['userId'] == user_id]['movieId'])
    unseen = set(movies['movieId']) - watched
    predictions = [(mid, model.predict(user_id, mid).est) for mid in unseen]
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_movie_ids = [p[0] for p in predictions[:topn]]
    return movies[movies['movieId'].isin(top_movie_ids)][['movieId', 'title']]

# --- Cold Start Hybrid with Mood ---
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

    user_profiles = {}
    for uid in ratings['userId'].unique():
        liked = ratings[(ratings['userId'] == uid) & (ratings['rating'] >= 4.0)]
        liked_ids = liked['movieId'].values
        liked_idx = movies[movies['movieId'].isin(liked_ids)].index
        if len(liked_idx) == 0:
            continue
        liked_vectors = soup_matrix[liked_idx]
        user_profiles[uid] = np.asarray(liked_vectors.mean(axis=0)).reshape(1, -1)

    similarities = {
        uid: cosine_similarity(profile_vector, vec)[0][0]
        for uid, vec in user_profiles.items()
    }

    similar_user_ids = [uid for uid, _ in sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:5]]

    movie_scores = {}
    for uid in similar_user_ids:
        seen = set(ratings[ratings['userId'] == uid]['movieId'])
        for mid in movies['movieId']:
            if mid in seen:
                continue
            pred = model.predict(uid, mid).est
            movie_scores.setdefault(mid, []).append(pred)

    for mid in movie_scores:
        movie_scores[mid] = np.mean(movie_scores[mid])

    content_sim = cosine_similarity(profile_vector, soup_matrix).flatten()
    content_scores = dict(zip(movies['movieId'], content_sim))

    hybrid_scores = {}
    for mid in movie_scores:
        if mid in content_scores:
            hybrid_scores[mid] = 0.5 * movie_scores[mid] + 0.5 * content_scores[mid]

    top_movie_ids = [mid for mid, _ in sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:topn]]
    return movies[movies['movieId'].isin(top_movie_ids)][['movieId', 'title']]
