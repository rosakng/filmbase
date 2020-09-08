from chalice import Chalice
import pandas as pd
from sklearn.metrics.pairwise import sigmoid_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
import urllib.parse

from utils.helper import get_serialized_films, get_weighted_rating, get_recommendations_by_title

app = Chalice(app_name='filmbase')


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


@app.route('/v1/filmbase/results', methods=["GET"])
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

    cosine_similarity = sigmoid_kernel(tfidf_matrix, tfidf_matrix)

    # Create a reverse map of movie titles and DataFrame indices.
    # This is so that we can find the index of a movie in our DataFrame, given its title.
    indices = pd.Series(movies.index, index=movies['original_title']).drop_duplicates()

    return get_recommendations_by_title(movies, title, indices, cosine_similarity)

