import pandas as pd
import numpy as np


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


def get_weighted_rating(movies):
    V = movies['vote_count']
    R = movies['vote_average']
    C = movies['vote_average'].mean()

    # Experiment and change the quantile
    m = movies['vote_count'].quantile(0.90)
    # Calculation based on the IMDB formula
    return (V / (V + m) * R) + (m / (m + V) * C)


def get_recommendations_by_title(movies, title, indices, cosine_similarity):
    index = indices[title]

    similarity_scores = list(enumerate(cosine_similarity[index]))

    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    top_ten = similarity_scores[1:11]
    movie_indices = [i[0] for i in top_ten]

    return str(movies['original_title'].iloc[movie_indices])


def get_director_name(data):
    for i in data:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


def get_top_elements(data):
    if isinstance(data, list):
        names = [i['name'] for i in data]

        if len(names) > 3:
            names = names[:3]
        return names
    return []
