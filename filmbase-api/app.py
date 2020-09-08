from chalice import Chalice

import pandas as pd
import numpy as np
import urllib.parse

from sklearn.metrics.pairwise import sigmoid_kernel, cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from ast import literal_eval

from utils.helper import get_serialized_films, get_recommendations_by_title

app = Chalice(app_name='filmbase')


@app.route('/v1/filmbase/test', methods=["GET"])
def tester():
    movies = get_serialized_films()
    return str(movies.head(2))


@app.route('/v1/filmbase/trending', methods=["GET"])
def get_trending_now():
    movies = get_serialized_films()

    V = movies['vote_count']
    R = movies['vote_average']
    C = movies['vote_average'].mean()
    # experiment and change the quantile
    m = movies['vote_count'].quantile(0.90)

    filtered_movies = movies.copy().loc[movies['vote_count'] >= m]
    filtered_movies['weighted_average'] = (V / (V + m) * R) + (m / (m + V) * C)

    filtered_movies = filtered_movies.sort_values('weighted_average', ascending=False)
    filtered_movies[['original_title', 'vote_count', 'vote_average', 'weighted_average', 'popularity']].head(20)
    return str(filtered_movies.head(10))


@app.route('/v1/filmbase/results/plot', methods=["GET"])
def get_reccs_by_plot():
    # EXAMPLE URL:
    # curl -X GET http://localhost:8000/v1/filmbase/results?search_query=The+Avengers
    #
    # Encoded title: The+Avengers
    # Decoded title: The Avengers

    encoded_title = app.current_request.query_params['search_query']
    title = urllib.parse.unquote_plus(encoded_title)

    movies = get_serialized_films()

    V = movies['vote_count']
    R = movies['vote_average']
    C = movies['vote_average'].mean()
    # experiment and change the quantile
    m = movies['vote_count'].quantile(0.70)

    movies['weighted_average'] = (V / (V + m) * R) + (m / (m + V) * C)

    tfidf = TfidfVectorizer(stop_words='english')

    # Replace NaN with an empty string
    movies['overview'] = movies['overview'].fillna('')
    tfidf_matrix = tfidf.fit_transform(movies['overview'])

    # in (X,Y), Y is the number of different words that were use to describe X movies in the data
    tfidf_matrix.shape

    cos_similarity = sigmoid_kernel(tfidf_matrix, tfidf_matrix)

    # Create a reverse map of movie titles and DataFrame indices.
    # This is so that we can find the index of a movie in our DataFrame, given its title.
    indices = pd.Series(movies.index, index=movies['original_title']).drop_duplicates()

    return get_recommendations_by_title(movies, title, indices, cos_similarity)


@app.route('/v1/filmbase/results/keyword', methods=["GET"])
def get_reccs_by_keywords():
    # EXAMPLE URL:
    # curl -X GET http://localhost:8000/v1/filmbase/results?search_query=The+Avengers
    #
    # Encoded title: The+Avengers
    # Decoded title: The Avengers

    encoded_title = app.current_request.query_params['search_query']
    title = urllib.parse.unquote_plus(encoded_title)

    movies = get_serialized_films()

    sections = ['cast', 'crew', 'keywords', 'genres']
    for section in sections:
        movies[section] = movies[section].apply(literal_eval)

    movies['director'] = movies['crew'].apply(get_director_name)

    features = ['cast', 'keywords', 'genres']
    for feature in features:
        movies[feature] = movies[feature].apply(get_top_elements)

    print(str(movies[['original_title', 'cast', 'director', 'keywords', 'genres']].head(3)))

    features_2 = ['cast', 'keywords', 'director', 'genres']

    for feature_2 in features_2:
        movies[feature_2] = movies[feature_2].apply(serialize)

    movies['soup'] = movies.apply(create_metadata_soup, axis=1)
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(movies['soup'])

    cos_similarity = cosine_similarity(count_matrix, count_matrix)

    movies = movies.reset_index()
    indices = pd.Series(movies.index, index=movies['original_title'])
    return get_recommendations_by_title(movies, title, indices, cos_similarity)


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


# This method is used to convert data to lowercase and strip all the spaces so that the vectorizer recognizes common
# words
def serialize(data):
    if isinstance(data, list):
        return [str.lower(i.replace(" ", "")) for i in data]
    else:
        if isinstance(data, str):
            return str.lower(data.replace(" ", ""))
        return ""


def create_metadata_soup(data):
    return ' '.join(data['keywords']) + ' ' + ' '.join(data['cast']) + ' ' + data['director'] + ' ' + ' '.join(
        data['genres'])
