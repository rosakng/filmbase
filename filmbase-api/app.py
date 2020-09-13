from chalice import Chalice

import pandas as pd
import urllib.parse

from sklearn.metrics.pairwise import sigmoid_kernel, cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from surprise.model_selection import cross_validate
from surprise import Reader, Dataset, SVD
from ast import literal_eval
from math import sqrt

from utils.helper import get_film_data, get_recommendations_by_title, get_weighted_rating, get_director_name, \
    get_top_elements, convert_data, create_metadata_soup, get_clean_movies_dataframe, get_clean_ratings_dataframe

app = Chalice(app_name='filmbase')


@app.route('/v1/filmbase/trending', methods=["GET"])
def get_trending_now():
    movies = get_film_data()

    # Experiment and change the quantile
    m = movies['vote_count'].quantile(0.90)

    print(m)

    filtered_movies = movies.copy().loc[movies['vote_count'] >= m]

    print(filtered_movies.shape)

    filtered_movies['weighted_average'] = get_weighted_rating(movies)

    filtered_movies = filtered_movies.sort_values('weighted_average', ascending=False)

    return str(
        filtered_movies[['original_title', 'vote_count', 'vote_average', 'weighted_average', 'popularity']].head(20))


@app.route('/v1/filmbase/results/plot', methods=["GET"])
def get_reccs_by_plot():
    # EXAMPLE URL:
    # curl -X GET http://localhost:8000/v1/filmbase/results?search_query=The+Avengers
    #
    # Encoded title: The+Avengers
    # Decoded title: The Avengers

    encoded_title = app.current_request.query_params['search_query']
    title = urllib.parse.unquote_plus(encoded_title)

    movies = get_film_data()

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


@app.route('/v1/filmbase/results/metadata', methods=["GET"])
def get_reccs_by_keywords_credits_genres():
    # This endpoint returns the recommendations based on the metadata: 3 top actors, director, related genres,
    # and movie plot keywords

    # EXAMPLE URL:
    # curl -X GET http://localhost:8000/v1/filmbase/results?search_query=The+Avengers
    #
    # Encoded title: The+Avengers
    # Decoded title: The Avengers

    encoded_title = app.current_request.query_params['search_query']
    title = urllib.parse.unquote_plus(encoded_title)

    movies = get_film_data()

    # These sections are the existing columns we'll extract data from
    sections = ['cast', 'crew', 'keywords', 'genres']
    for section in sections:
        # We use ast.literal_eval to parse the "stringified" list data into usable python objects
        movies[section] = movies[section].apply(literal_eval)

    # Define director column with corresponding movie directors from the crew section
    movies['director'] = movies['crew'].apply(get_director_name)

    columns = ['cast', 'keywords', 'genres']
    for column in columns:
        movies[column] = movies[column].apply(get_top_elements)

    # show snippet for new columns
    # print(str(movies[['original_title', 'cast', 'director', 'keywords', 'genres']].head(3)))

    features = ['cast', 'keywords', 'director', 'genres']
    for feature in features:
        # parse data in feature columns
        movies[feature] = movies[feature].apply(convert_data)

    # show snippet for serialized data
    # print(str(movies[['original_title', 'cast', 'director', 'keywords', 'genres']].head(3)))

    movies['soup'] = movies.apply(create_metadata_soup, axis=1)
    # print(str(movies['soup'].head(3)))

    # We use CountVectorizer here instead of TF-IDF so that we can account for actors/directors that have been a part
    # of more than 1 movie
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(movies['soup'])

    # Get cosine similarity matrix
    cos_similarity = cosine_similarity(count_matrix, count_matrix)

    # Reset main DataFrame and construct reverse mapping
    movies = movies.reset_index()
    indices = pd.Series(movies.index, index=movies['original_title'])
    return get_recommendations_by_title(movies, title, indices, cos_similarity)


@app.route('/v1/filmbase/results/ratings', methods=["GET"])
def get_reccs_by_ratings():
    reader = Reader()
    # Use a new data set from movie lens that contain userId
    ratings = pd.read_csv("data/ratings_small.csv")

    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    svd = SVD()

    # See that the Root Mean Square Error is approx. 0.89
    print(cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5))

    # training data set
    train_set = data.build_full_trainset()
    svd.fit(train_set)

    # Show ratings that userId 1 has made
    print(ratings[ratings['userId'] == 1])

    # For svd.predict(X, Y, Z) Return prediction that for userID X with movieId Y
    # Pick arbitrary true rating Z (optional)
    return svd.predict(1, 100, 3)


@app.route('/v1/filmbase/results/reccs', methods=["GET"])
def get_reccs_by_user_input_ratings():
    # This method uses Collaborative Filtering (User-User Filtering) to generate recommended items
    # It attempts to find users that have similar preferences by using the Pearson Correlation Function
    movies_df = get_clean_movies_dataframe()
    ratings_df = get_clean_ratings_dataframe()

    # TODO: Create post request for user input
    userInput = [
        {'title': 'Breakfast Club, The', 'rating': 5},
        {'title': 'Toy Story', 'rating': 3.5},
        {'title': 'Jumanji', 'rating': 2},
        {'title': "Pulp Fiction", 'rating': 5},
        {'title': 'Akira', 'rating': 4.5}
    ]
    # Generate dataframe for user input with rated movies
    inputMovies = pd.DataFrame(userInput)
    # print(inputMovies.head())

    # Find rows that contain the rated movie titles, then merge it
    # Merged dataframe will movieId, title, and rating of movies that were rated
    inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
    # TODO: some check to make sure that the rated movies are in the movie dataframe
    inputMovies = pd.merge(inputId, inputMovies)
    inputMovies = inputMovies.drop('year', 1)

    # Use ratings dataframe and get users that have rated the same movies as the input
    userSubset = ratings_df[ratings_df['movieId'].isin(inputMovies['movieId'].tolist())]
    # print(userSubset.head())
    userSubsetGroup = userSubset.groupby(['userId'])
    # Group by a certain user id:
    # print(userSubsetGroup.get_group(200))
    userSubsetGroup = sorted(userSubsetGroup, key=lambda x: len(x[1]), reverse=True)
    # print(userSubsetGroup[:3])

    userSubsetGroup = userSubsetGroup[0:100]

    # use Pearson Correlation Coefficient to find similarity of users to the input user
    pearsonCorrelationDict = {}
    # For every user group in our subset
    for name, group in userSubsetGroup:
        # Sorting the input and current user group
        group = group.sort_values(by='movieId')
        inputMovies = inputMovies.sort_values(by='movieId')
        # N value for the formula
        nRatings = len(group)
        # Review scores for the movies that they both have in common
        temp_df = inputMovies[inputMovies['movieId'].isin(group['movieId'].tolist())]
        # And then store them in a temporary buffer variable in a list
        tempRatingList = temp_df['rating'].tolist()
        # Put the current user group reviews in a list format
        tempGroupList = group['rating'].tolist()

        # Calculate the pearson correlation between two users, so called, x and y
        # See Sum of Squares formula
        Sxx = sum([i ** 2 for i in tempRatingList]) - pow(sum(tempRatingList), 2) / float(nRatings)
        Syy = sum([i ** 2 for i in tempGroupList]) - pow(sum(tempGroupList), 2) / float(nRatings)
        Sxy = sum(i * j for i, j in zip(tempRatingList, tempGroupList)) - sum(tempRatingList) * sum(
            tempGroupList) / float(nRatings)

        # If the denominator is not zero, then divide, else, set correlation to 0
        if Sxx != 0 and Syy != 0:
            pearsonCorrelationDict[name] = Sxy / sqrt(Sxx * Syy)
        else:
            pearsonCorrelationDict[name] = 0

        # print(pearsonCorrelationDict.items())

        pearsonDF = pd.DataFrame.from_dict(pearsonCorrelationDict, orient='index')
        pearsonDF.columns = ['similarityIndex']
        pearsonDF['userId'] = pearsonDF.index
        pearsonDF.index = range(len(pearsonDF))

        # Change this number to choose top X about of similar users
        topUsers = pearsonDF.sort_values(by='similarityIndex', ascending=False)[0:60]

        # Get movies watched by the users in pearson datafram from the ratings dataframe and store in "similarityIndex"
        # Take weighted average of the ratings of movies using Pearson Correlation as the weight
        topUsersRating = topUsers.merge(ratings_df, left_on='userId', right_on='userId', how='inner')
        topUsersRating['weightedRating'] = topUsersRating['similarityIndex'] * topUsersRating['rating']

        # Applies a sum to the topUsers for similarity index and weighted rating
        tempTopUsersRating = topUsersRating.groupby('movieId').sum()[['similarityIndex', 'weightedRating']]
        tempTopUsersRating.columns = ['sum_similarityIndex', 'sum_weightedRating']

        # Empty dataframe
        recommendation_df = pd.DataFrame()
        # Take the weighted average and add column for it
        recommendation_df['weighted average recommendation score'] = tempTopUsersRating['sum_weightedRating'] / tempTopUsersRating['sum_similarityIndex']
        recommendation_df['movieId'] = tempTopUsersRating.index

        # Sort dataframe and see top X amount of movies
        recommendation_df = recommendation_df.sort_values(by='weighted average recommendation score', ascending=False)
        return str(movies_df.loc[movies_df['movieId'].isin(recommendation_df.head(10)['movieId'].tolist())])


if __name__ == '__main__':
    get_reccs_by_user_input_ratings()
