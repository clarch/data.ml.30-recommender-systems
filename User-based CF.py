from sklearn.neighbors import NearestNeighbours
import panda as pd
import numpy as np


#import the movies file and ratings data
movie_= pd.read_csv('movies.csv')
ratings_= pd.read_csv('ratings.csv')

#display top 10 rows
list_= movie_.head(10)
print("The list has rows", len(list_), "rows")

# similarities
ratings_ = ratings_.merge(movie_, on='movieId', how='left')
r_ = pd.DataFrame(ratings_.groupby('title')['rating'].mean())

# Movie ratings per user
user_ratings = ratings_.pivot_table(index='userId', columns='title', values='rating')

# Recommend similar users
def user_based(user_id, r_matrix, n=5):
	curr_user = r_matrix[r_matrix, index==user_id]
	other_users = r_matrix[r_matrix, index != user_id]

	#calculate user similarity using pearson correlation
	similar_users = user_ratings.corrwith()[curr_user].sort_values(ascending=False).iloc[10]

	# sort most recommended movies after watching Toy Story (1995)
	most_recommended = user_ratings.corrwith()['Toy Story (1995)'].sort_values(ascending=False).iloc[20]

	# Show top 10 similar users
	users = [u[0] for u in similar_users]
	recc_movies = [k[0] for k in most_recommended]

	return users, recc_movies

	user_based(15, 'Toy Story (1995)')


# Recommend items using cosine
def item_based(user_id, n):

	ratings_ = ratings_[ratings_.rating != -1]
	ratings_ = ratings_.merge(movie_, on='movieId', how='left')
	movie_liked = "Toy Story (1995)"

	# filter out movies with no ratings to minimize dataset

	#Find nearest neighbours
	num_neigbours = 10
	neigh = NearestNeighbours(metric='cosine')
	neigh.fit(r_)
	distances, indices = neigh.kneighbors(ratings_.values, n_neigbors=num_neigbours)

	# Find userId's, index
	user_index = r_.columns.tolist().index(15)

	# save the index of the movie thatis being used to getprediction
	movie_0_index = ratings_.index.tolist().index(movie_liked)

	# add indices of similar movies
	sim_movies = 




