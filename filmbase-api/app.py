from chalice import Chalice
import pandas as pd
from sklearn.metrics.pairwise import sigmoid_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


from utils.helper import get_serialized_films, get_weighted_rating

app = Chalice(app_name='filmbase')


@app.route('/v1', methods=["GET"])
def index():
    return get_recs('Mean Girls')

#  WIP
@app.route('/v1/filmbase', methods=["GET"])
def get_recs(title):
    movies = get_serialized_films()

    V = movies['vote_count']
    R = movies['vote_average']
    C = movies['vote_average'].mean()
    # experiment and change the quantile
    m = movies['vote_count'].quantile(0.70)

    movies['weighted_average'] = (V / (V + m) * R) + (m / (m + V) * C)

    # Using Abhishek Thakur's arguments for TF-IDF
    tfv = TfidfVectorizer(min_df=3, max_features=None,
                          strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                          ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,
                          stop_words='english')

    # Filling NaNs with empty string
    movies['overview'] = movies['overview'].fillna('')

    # Fitting the TF-IDF on the 'overview' text
    tfv_matrix = tfv.fit_transform(movies['overview'])

    # tfv_matrix.shape

    # Compute the sigmoid kernel
    sig = sigmoid_kernel(tfv_matrix, tfv_matrix)

    # Reverse mapping of indices and movie titles
    indices = pd.Series(movies.index, index=movies['original_title']).drop_duplicates()

    # Get the index corresponding to original_title
    idx = indices[title]

    # Get the pairwsie similarity scores
    sig_scores = list(enumerate(sig[idx]))

    # Sort the movies
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)

    # Scores of the 10 most similar movies
    sig_scores = sig_scores[1:11]

    # Movie indices
    movie_indices = [i[0] for i in sig_scores]

    # Top 10 most similar movies
    return str(movies['original_title'].iloc[movie_indices])


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


@app.route('/v1/filmbase/plot', methods=["GET"])
def get_reccs_by_plot():
    title = app.current_request.query_params["title"]

    movies = get_serialized_films()
    tfidf = TfidfVectorizer(stop_words='english')

    # Replace NaN with an empty string
    movies['overview'] = movies['overview'].fillna('')
    tfidf_matrix = tfidf.fit_transform(movies['overview'])

    # in (X,Y), Y is the number of different words that were use to describe X movies in the data
    tfidf_matrix.shape

    cosine_similarity = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Create a reverse map of movie titles and DataFrame indices.
    # This is so that we can find the index of a movie in our DataFrame, given its title.
    indices = pd.Series(tfidf_matrix.index, index=tfidf_matrix['title']).drop_duplicates()

    return get_recommendations_by_title(movies, title, indices, cosine_similarity)


def get_recommendations_by_title(movies, title, indices, cosine_similarity):
    index = indices[title]

    similarity_scores = list(enumerate(cosine_similarity[index]))

    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    top_ten = similarity_scores[1:11]
    movie_indices = [i[0] for i in top_ten]

    return movies['title'].iloc[movie_indices]

