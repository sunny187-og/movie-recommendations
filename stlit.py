from proj import movies, tfidf, soup_matrix, model, cold_start_hybrid_with_mood


import streamlit as st
import pandas as pd

# --------------------------
# Load Data & Models
# --------------------------
# You must load these externally and pass them to this file:
# - movies: your DataFrame with 'title', 'movieId', 'soup'
# - tfidf, soup_matrix: TF-IDF vectorizer and matrix
# - model: trained Surprise SVD model
# - hybrid function: cold_start_hybrid_with_mood()

# For demo/testing only:
# from your_module import movies, tfidf, soup_matrix, model, cold_start_hybrid_with_mood

# --------------------------
# Page UI
# --------------------------
st.set_page_config(page_title="Movie Recommender", layout="centered")
st.title("🎬 Hybrid Movie Recommendation System")
st.markdown("Personalized movie suggestions based on your favorites and your mood")

# --------------------------
# Collect Cold Start Inputs
# --------------------------
st.header("Tell us about your taste")
fav_genres = st.multiselect("Choose your favorite genres:",
                            ['Action', 'Adventure', 'Comedy', 'Drama', 'Fantasy', 'Horror', 'Romance', 'Sci-Fi', 'Thriller'])
fav_movie = st.text_input("Favorite movie (optional):")
fav_actor = st.text_input("Favorite actor (optional):")
fav_director = st.text_input("Favorite director (optional):")

# --------------------------
# Mood Input
# --------------------------
st.header("How are you feeling today?")
mood = st.selectbox("Select your current mood:",
                    ['happy', 'sad', 'angry', 'bored', 'romantic', 'excited', 'scared', 'inspired'])

# --------------------------
# Get Recommendations
# --------------------------
if st.button("🎥 Recommend Movies"):
    if not fav_genres and not fav_movie and not fav_actor and not fav_director:
        st.warning("Please provide at least one preference.")
    else:
        with st.spinner("Generating recommendations..."):
            try:
                recommendations = cold_start_hybrid_with_mood(
                    fav_genres=fav_genres,
                    fav_movie=fav_movie,
                    fav_actor=fav_actor,
                    fav_director=fav_director,
                    mood=mood,
                    topn=10
                )
                st.success("Here are your recommendations:")
                st.dataframe(recommendations.reset_index(drop=True))
            except Exception as e:
                st.error(f"Error generating recommendations: {e}")
