from chalice import Chalice

import pandas as pd
import numpy as np
import urllib.parse

from sklearn.metrics.pairwise import sigmoid_kernel, cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from surprise.model_selection import cross_validate
from surprise import Reader, Dataset, SVD
from ast import literal_eval

from utils.helper import get_serialized_films, get_recommendations_by_title, get_weighted_rating, get_director_name, \
    get_top_elements

app = Chalice(app_name='filmbase')


@app.route('/v1/filmbase/trending', methods=["GET"])
def get_trending_now():
    movies = get_serialized_films()

    # Experiment and change the quantile
    m = movies['vote_count'].quantile(0.90)

    print(m)

    filtered_movies = movies.copy().loc[movies['vote_count'] >= m]

    print(filtered_movies.shape)

    filtered_movies['weighted_average'] = get_weighted_rating(movies)

    filtered_movies = filtered_movies.sort_values('weighted_average', ascending=False)

    return str(filtered_movies[['original_title', 'vote_count', 'vote_average', 'weighted_average', 'popularity']].head(20))


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

    tf_idf = TfidfVectorizer(stop_words='english')

    # Replace NaN with an empty string
    movies['overview'] = movies['overview'].fillna('')
    tf_idf_matrix = tf_idf.fit_transform(movies['overview'])

    # in (X,Y), Y is the number of different words that were use to describe X movies in the data
    print(tf_idf_matrix.shape)

    cos_similarity = sigmoid_kernel(tf_idf_matrix, tf_idf_matrix)

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


@app.route('/v1', methods=["GET"])
def collab_filter():
    reader = Reader()
    ratings = pd.read_csv("data/ratings_small.csv")
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

    svd = SVD()
    print(cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5))

    trainset = data.build_full_trainset()
    svd.fit(trainset)

    print(ratings[ratings['userId'] == 1])

    return svd.predict(1, 302, 3)


if __name__ == '__main__':
    collab_filter()