#from sklearn.neighbors import NearestNeighbours
import pandas as pd
import numpy as np


#import the movies file and ratings data
movies_list = pd.read_csv('../excel/ml-latest-small/movies.csv')
ratings_list = pd.read_csv('../excel/ml-latest-small/ratings.csv')

#display top 10 rows
mov_list = movies_list.head(10)
print(mov_list)
print("Number of elements in the dataset:", len(ratings_list))

# merge ratings list with movie list
ratings_list = ratings_list.merge(movies_list, on='movieId', how='left')

# create dataframe with movie titles and their ratings
ratings_df = pd.DataFrame(ratings_list.groupby('title')['rating'].mean())

#corr = user_ratings.corrwith(user_ratings["GoldenEye (1995)"])
#corr= corr.head(10)
#print("movies", corr)

# Recommend similar users
def user_based(user_id, r_matrix, movieName, n=5):

		# create a list with all the users and their ratings for movies
		user_ratings = r_matrix.pivot_table(index='userId', columns='title', values='rating')
		user_ratings = user_ratings.fillna(0)
		
		# Get similar users
				
		# Similar movies to movieName, recommend with pearsons correlation
		correlation = user_ratings.corrwith(user_ratings[movieName]).sort_values(ascending=False)
		correlation = correlation.head(20)
		print(correlation)

user_based(100, ratings_list, 'Enemy of the State (1998)')


# Recommend items using cosine
#def item_based(user_id, n):
#
#	ratings_ = ratings_[ratings_.rating != -1]
#	ratings_ = ratings_.merge(movie_, on='movieId', how='left')
#	movie_liked = "Toy Story (1995)"
#	# filter out movies with no ratings to minimize dataset

	#Find nearest neighbours
#	num_neigbours = 10
#	neigh.fit(r_)
#
	# Find userId's, index
#	user_index = r_.columns.tolist().index(15)

	# save the index of the movie thatis being used to getprediction
#
	# add indices of similar movies
#	sim_movies = 




