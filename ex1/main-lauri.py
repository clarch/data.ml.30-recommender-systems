import pandas as pd
from math import sqrt
import os

TARGET_USER = 1
TARGET_MOVIE = 2

# Check the working directory
print('Current working directory: ', os.getcwd())

# Import ratings data and print first 5 elements of it
ratings = pd.read_csv("./dataset/ml-latest-small/ratings.csv")
print('First five elements of this dataset:')
print(ratings.head(5))

print('Number of elements in dataset: ', len(ratings.index), "\n")

# Returns rating given by userId to movie movieId
def getUsersRatingForMovie(all_ratings, userId, movieId):
    rating = all_ratings[(all_ratings.userId == userId) & (all_ratings.movieId == movieId)]
    if (len(rating.index) == 0):
        return -1
    return rating.rating.iloc[0]


# Returns all rating given by userId
def getUsersRatings(all_ratings, userId):
    return all_ratings.loc[all_ratings.userId == userId].movieId.tolist()

# Returns mean score for all ratings given by userId
def getUsersRatingMean(all_ratings, userId):
    usersRatings = all_ratings.loc[all_ratings.userId == userId]
    return usersRatings.rating.mean()

# Calculates sum of users nomalized ratings in a dataset
def calculateSumOfNormalizedRatingsCubed(allRatings, usersRatings, userId):
    usersMeanRating = getUsersRatingMean(allRatings, userId)
    sumOfNormalizedRatings = 0
    for i, row in usersRatings.iterrows():
        sumOfNormalizedRatings += ((row.rating - usersMeanRating) ** 2) 
    return sqrt(sumOfNormalizedRatings)

# Calculate pearsons correlation between given users on a dataset of ratings
# Excpects dataset (DataFrame object) to have columns 'userId', 'movieId', 'rating'
# Based on formula in Falk, K. (2019). Practical Recommender Systems. Manning Publications Co. LLC.
# minCommonRating: If this attribute is given, we will exclude those from this calculation
def calculatePearsonsCorrelationBetweenUsers(ratings, user_a_id, user_b_id, minCommonRatings = 0):
    # Get a dataframe of all movies user a has rated and a simple list of all the id's
    a_ratings = ratings.loc[ratings.userId == user_a_id]
    b_ratings = ratings.loc[ratings.userId == user_b_id]

    a_rating_movies_list = a_ratings.movieId.values.tolist()
    b_rating_movies_list = b_ratings.movieId.values.tolist()

    # Create two dataframes, which both contain only movies that both users have rarated
    a_union_b_boolean_series = a_ratings.movieId.isin(b_rating_movies_list)
    b_union_a_boolean_series = b_ratings.movieId.isin(a_rating_movies_list)

    setA = a_ratings[a_union_b_boolean_series].sort_values(by=['movieId'])
    setB = b_ratings[b_union_a_boolean_series].sort_values(by=['movieId'])

    # Check that users have common movies rated
    # print("Count of commont ratings: ", len(setA.index))
    if (len(setA.index) == 0):
        print("Users have no reviews on same movies. No similarity score can be calculated")
        return 0

    # If users have less common ratings than given limit
    if (len(setA.index) < minCommonRatings):
        return 0

    combinedSet = pd.merge(
        setA.rename(columns={'rating': 'rating_a'})[['movieId', 'rating_a']], 
        setB.rename(columns={'rating': 'rating_b'})[['movieId', 'rating_b']],
        on = 'movieId')[['rating_a', 'rating_b']]

    sumOfNormilizedMultipliedRatings = 0
    for i, row in combinedSet.iterrows():
        a_norm = row.rating_a - getUsersRatingMean(ratings, user_a_id)
        b_norm = row.rating_b - getUsersRatingMean(ratings, user_b_id)
        sumOfNormilizedMultipliedRatings += (a_norm * b_norm)

    divider = calculateSumOfNormalizedRatingsCubed(ratings, setA, user_a_id) * calculateSumOfNormalizedRatingsCubed(ratings, setB, user_b_id)
    
    if divider == 0:
        return 0
    
    return sumOfNormilizedMultipliedRatings / divider



correlation = calculatePearsonsCorrelationBetweenUsers(ratings, 1, 2)
print("Pearson's correlation between user A nd B: ", correlation)

# Finds the most similar users to target user that have rated given target movie
def findMostSimilarUsers(ratings, targetUser, numberOfUsers, targetMovie, minCommonRatings = 0, usersLimit = 0):
    allUserIds = ratings.userId.unique().tolist()

    if usersLimit == 0:
        usersLimit = len(allUserIds)

    targetUsersRating = getUsersRatingForMovie(ratings, targetUser, targetMovie)
    if targetUsersRating >= 0:
        print("Target user has already rated target movie: ", targetMovie, " with rating: ", targetUsersRating)
        return -1
    
    # Getting similarity score between targeted and every other user. Tupple of (similarityScore, userId) is returned
    # Is only done for users, that have rated the target movie
    similarityScores = [(calculatePearsonsCorrelationBetweenUsers(ratings, targetUser, anotherUser, minCommonRatings), anotherUser) \
        for anotherUser in allUserIds[:usersLimit] \
            if (anotherUser != targetUser) & (getUsersRatingForMovie(ratings, anotherUser, targetMovie) >= 0)]
    
    # Sort results and return first n number of users
    similarityScores.sort(reverse=True) # ### ## # ## 
    return similarityScores[:numberOfUsers]


def predictUsersRating(ratings, targetUser, targetMovie):
    similarUsers = findMostSimilarUsers(ratings, targetUser, 10, targetMovie, 10, 30)

    if similarUsers == -1:
        return -1

    targetUsersRatingMean = getUsersRatingMean(ratings, targetUser)

    sumOfSimilarUsersWeightedNormilizedRatings = 0
    for similarUser in similarUsers:
        usersNormalizedRating = getUsersRatingForMovie(ratings, similarUser[1], targetMovie) - getUsersRatingMean(ratings, similarUser[1])
        sumOfSimilarUsersWeightedNormilizedRatings += similarUser[0] * usersNormalizedRating

    sumOfSimilarityScores = sum([user[0] for user in similarUsers])
    
    return targetUsersRatingMean + (sumOfSimilarUsersWeightedNormilizedRatings / sumOfSimilarityScores)


predictedRating = predictUsersRating(ratings, TARGET_USER, TARGET_MOVIE)
print('Target user: ', TARGET_USER, '\nTarget movie: ', TARGET_MOVIE, '\nPredicted rating: ', predictedRating)