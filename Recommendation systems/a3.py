# coding: utf-8

# # Assignment 3:  Recommendation systems
#
# Here we'll implement a content-based recommendation algorithm.
# It will use the list of genres for a movie as the content.
# The data come from the MovieLens project: http://grouplens.org/datasets/movielens/

# Please only use these imports.
from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile

def download_data():
    """ DONE. Download and unzip data.
    """
    url = 'https://www.dropbox.com/s/h9ubx22ftdkyvd5/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()


def tokenize_string(my_string):
    """ DONE. You should use this in your tokenize function.
    """
    return re.findall('[\w\-]+', my_string.lower())


def tokenize(movies):
    """
    Append a new column to the movies DataFrame with header 'tokens'.
    This will contain a list of strings, one per token, extracted
    from the 'genre' field of each movie. Use the tokenize_string method above.

    Note: you may modify the movies parameter directly; no need to make
    a new copy.
    Params:
      movies...The movies DataFrame
    Returns:
      The movies DataFrame, augmented to include a new column called 'tokens'.

    >>> movies = pd.DataFrame([[123, 'Horror|Romance'], [456, 'Sci-Fi']], columns=['movieId', 'genres'])
    >>> movies = tokenize(movies)
    >>> movies['tokens'].tolist()
    [['horror', 'romance'], ['sci-fi']]
    """
    movie_tokens =()
    
    for genres in movies['genres']:
        movie_tokens += (tokenize_string(genres),)
    
    movies['tokens']=list(movie_tokens)
    
    return movies
    pass


def featurize(movies):
    """
    Append a new column to the movies DataFrame with header 'features'.
    Each row will contain a csr_matrix of shape (1, num_features). Each
    entry in this matrix will contain the tf-idf value of the term, as
    defined in class:
    tfidf(i, d) := tf(i, d) / max_k tf(k, d) * log10(N/df(i))
    where:
    i is a term
    d is a document (movie)
    tf(i, d) is the frequency of term i in document d
    max_k tf(k, d) is the maximum frequency of any term in document d
    N is the number of documents (movies)
    df(i) is the number of unique documents containing term i

    Params:
      movies...The movies DataFrame
    Returns:
      A tuple containing:
      - The movies DataFrame, which has been modified to include a column named 'features'.
      - The vocab, a dict from term to int. Make sure the vocab is sorted alphabetically as in a2 (e.g., {'aardvark': 0, 'boy': 1, ...})
    """
    vocab_tuples=()
    for tokens_list in movies['tokens']:
        for tokens in tokens_list:
            vocab_tuples += (tokens,)
    
    vocab ={}
    for vocab_elements in sorted(set(list(vocab_tuples))):
         vocab[vocab_elements]=len(vocab)

    
    Number_of_docs=()
    for documents in movies['tokens']:
        Number_of_docs += (documents,)
        
    N = len(list(Number_of_docs))
    
    Unique = {}
    for term in vocab.keys():
        genre_count = 0
        for documents in movies['tokens']:
            if term in documents:
                genre_count = genre_count + 1
        Unique[term]=genre_count
     
    features_list = ()
    Row_location = 0
    for document in movies['tokens']:
        csr_data_list = []
        csr_indices_list = []
        csr_indptr_list = []
        for terms in document:
            Document_Counters = Counter(document)
            max_terms= Document_Counters.most_common(1)
            doc_maximum = max_terms[0][1]
            Ndf = N/Unique[terms]
            tfidf = (Document_Counters[terms]/doc_maximum) * (math.log(Ndf)/math.log(10))
            csr_data_list.append(tfidf)
            csr_indices_list.append(Row_location)
            csr_indptr_list.append(vocab[terms])
            data_array = np.array(csr_data_list,np.float64)
            indices_array = np.array(csr_indices_list,np.int64)
            indptr_array = np.array(csr_indptr_list,np.int64)
        features_list += (csr_matrix((data_array, (indices_array, indptr_array)), shape=(1,len(vocab))),)
            
    movies['features']=list(features_list)
    
    return movies,vocab
    pass


def train_test_split(ratings):
    """DONE.
    Returns a random split of the ratings matrix into a training and testing set.
    """
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]


def cosine_sim(a, b):
    """
    Compute the cosine similarity between two 1-d csr_matrices.
    Each matrix represents the tf-idf feature vector of a movie.
    Params:
      a...A csr_matrix with shape (1, number_features)
      b...A csr_matrix with shape (1, number_features)
    Returns:
      The cosine similarity, defined as: dot(a, b) / ||a|| * ||b||
      where ||a|| indicates the Euclidean norm (aka L2 norm) of vector a.
    """
    dot_product_matrices = np.dot(a,b.transpose())
    
    square_values_a = 0
    for elements_a in a.data:
        squares_a = pow(elements_a,2)
        square_values_a = square_values_a + squares_a
    normalized_value_a= square_values_a**(1/2.0)
    
    square_values_b = 0
    for elements_b in b.data:
        squares_b = pow(elements_b,2)
        square_values_b = square_values_b + squares_b
    normalized_value_b= square_values_b**(1/2.0)
    
    normalized_product = normalized_value_a * normalized_value_b
    cosine_similarity = dot_product_matrices/normalized_product
    
    return cosine_similarity
    pass


def make_predictions(movies, ratings_train, ratings_test):
    """
    Using the ratings in ratings_train, predict the ratings for each
    row in ratings_test.

    To predict the rating of user u for movie i: Compute the weighted average
    rating for every other movie that u has rated.  Restrict this weighted
    average to movies that have a positive cosine similarity with movie
    i. The weight for movie m corresponds to the cosine similarity between m
    and i.

    If there are no other movies with positive cosine similarity to use in the
    prediction, use the mean rating of the target user in ratings_train as the
    prediction.

    Params:
      movies..........The movies DataFrame.
      ratings_train...The subset of ratings used for making predictions. These are the "historical" data.
      ratings_test....The subset of ratings that need to predicted. These are the "future" data.
    Returns:
      A numpy array containing one predicted rating for each element of ratings_test.
    """
    predicted_Rating=()
    for irow, row in ratings_test.iterrows():
        test_user = row['userId']
        testing_movie = movies.features[movies[movies.movieId==row['movieId']].index[0]]
        training_user_data=ratings_train.loc[ratings_train['userId'] == test_user]
        ratings_list,cosines,weighted_ratings =(),[],[]
        for jcol,col in training_user_data.iterrows():
            training_movies = movies.features[movies[movies.movieId==col['movieId']].index[0]]
            cos_sim = cosine_sim(training_movies,testing_movie)
            ratings_list += (col['rating'],)
            if(cos_sim>0):
                cosines.append(cos_sim)
                weighted_ratings.append(cos_sim*col['rating'])
        if(len(cosines)!=0):
            predicted_Rating += (sum(weighted_ratings)/sum(cosines),) 
        else:
            predicted_Rating += (np.mean(list(ratings_list)),)
             
    return np.array(list(predicted_Rating))
    pass


def mean_absolute_error(predictions, ratings_test):
    """DONE.
    Return the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()


def main():
    download_data()
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])


if __name__ == '__main__':
    main()
