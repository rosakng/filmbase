import pandas as pd
from sklearn.metrics.pairwise import sigmoid_kernel
from sklearn.feature_extraction.text import TfidfVectorizer


def get_serialized_films():
    credits = pd.read_csv("data/tmdb_5000_credits.csv")
    movies_incomplete = pd.read_csv("data/tmdb_5000_movies.csv")

    credits_renamed = credits.rename(index=str, columns={"movie_id": "id"})
    # join 2 datasets on the id column
    movies_dirty = movies_incomplete.merge(credits_renamed, on='id')
    print(movies_dirty.head())

    movies_clean = movies_dirty.drop(columns=['homepage', 'title_x', 'title_y', 'status', 'production_countries'])
    print(movies_clean.head())
    return movies_clean


def get_weighted_rating(movies, m, C):
    V = movies['vote_count']
    R = movies['vote_average']
    # Calculation based on the IMDB formula
    return (V / (V + m) * R) + (m / (m + V) * C)
