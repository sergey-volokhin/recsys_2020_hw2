from sklearn.neighbors import NearestNeighbors as NN
from scipy.stats import pearsonr

from collections import defaultdict
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import argparse


class fasterCF:

    def __init__(self, user_based, shrink_term=1):
        self.user_based = user_based
        self.shrink_term = shrink_term
        self.dataset = None
        self.NN = NN(metric='correlation')

    def fit(self, df):
        '''
            takes in pandas dataframe with columns ['critic_id', 'movie_id', 'score']
            constructs the NN model with correlation distance
        '''

        if self.user_based:
            index_on = 'critic_id'
            columns = 'movie_id'
        else:
            index_on = 'movie_id'
            columns = 'critic_id'
        self.dataset = df.pivot(index=index_on, columns=columns, values='score').fillna(0)
        self.NN.fit(self.dataset)

    def get_score(self, user_id, movie_id, neighbors):
        '''
            getting a score for a given user for a given movie, using the similarity matrix

            using a formula from surprise package (https://surprise.readthedocs.io/en/stable/knn_inspired.html, KNNWithMeans):

                            sum_{v in people} (sim(u, v) * (r_{vi} - mu_v))
            r_{ui} = mu_u + -----------------------------------------------   for user-to-user, and
                                    c + sum_{v in people} sim(u, v)


                            sum_{j in items} (sim(i, j) * (r_{uj} - mu_j))
            r_{ui} = mu_i + -----------------------------------------------   for item-to-item recommendations
                                    c + sum_{j in items} sim(i, j)

            where mu_i, mu_u, mu_j, and mu_v are mean ratings for particular item/user
        '''
        numerator = 0
        denominator = self.shrink_term
        if self.user_based:
            subdf = self.dataset.loc[user_id]  # getting all movies that critic left score for
            neighbors_dataset = self.dataset.loc[neighbors]
            working_series = neighbors_dataset[movie_id]
            for v in neighbors:
                v_series = neighbors_dataset.loc[v]
                mu_v = v_series[v_series != 0].mean()
                r_vi = working_series[v]
                if r_vi == 0:
                    continue
                sim_uv = pearsonr(v_series, subdf)[0]
                numerator += sim_uv * (r_vi - mu_v)
                denominator += sim_uv
        else:
            subdf = self.dataset.loc[movie_id]  # getting all critics who left score for that movie
            working_series = self.dataset[user_id]
            item_neighbors = self.NN.kneighbors([self.dataset.loc[movie_id]], n_neighbors=11)
            item_neighbors = list(self.dataset.iloc[item_neighbors[1][0][1:]].index)
            for j in item_neighbors:
                j_series = self.dataset.loc[j]
                mu_j = j_series[j_series != 0].mean()
                r_uj = working_series[j]
                if r_uj == 0:
                    continue
                sim_ij = pearsonr(j_series, subdf)[0]
                numerator += sim_ij * (r_uj - mu_j)
                denominator += sim_ij

        mu = subdf[subdf != 0].mean()  # mu_u/mu_i
        return mu + numerator / denominator

    def recommend(self, user_id, k):
        ''' recommendation function: gets scores for all (or closest to top-ranked by user) movies and ranks them, giving out top k of them '''
        if self.user_based:
            subdf = self.dataset.loc[user_id]
        else:
            subdf = self.dataset[user_id]
        scored_list = []
        print('Ranking movies')
        # if user-based, then get user neighbors and rank every movie for those neigbors
        if self.user_based:
            movies_pool = list(subdf[subdf == 0].index)
            neighbors = self.NN.kneighbors([self.dataset.loc[user_id]], n_neighbors=11)  # 11 is a magic number
            neighbors = list(self.dataset.iloc[neighbors[1][0][1:]].index)
        # if item-based, get most likely movies (by taking nearest neighbors for ones user rated highest) and rank them for all users
        else:
            movies_pool = set()
            user_df = self.dataset[user_id]
            top_ranked_movies = user_df[user_df == user_df.max()].index
            for movie in top_ranked_movies:
                movie_neighbors = self.NN.kneighbors([self.dataset.loc[movie]], n_neighbors=3)  # 3 is a magic number
                movies_pool.update(set(self.dataset.iloc[movie_neighbors[1][0][1:]].index))
            neighbors = 'all'
        for movie_id in tqdm(movies_pool):
            scored_list.append((movie_id, self.get_score(user_id, movie_id, neighbors)))
        scored_list.sort(key=lambda x: x[1], reverse=True)
        return scored_list[:k]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('system', choices=['user', 'item'])
    parser.add_argument('--amount-recommendations', '-k', default=10)
    args = parser.parse_args()

    if args.system=='user':
        model = fasterCF(user_based=True)
        print('USER BASED')
    else:
        model = fasterCF(user_based=False)
        print('ITEM BASED')

    print('\nLoading data')
    df_reviews = pd.read_csv(datapath+'reviews.tsv', sep='\t')
    popularity_thres = 50

    prev_shape = df_reviews.shape
    new_shape = 0
    df_med = df_reviews

    # removing critics and movies with less than 50 reviews (until convergence)
    while prev_shape != new_shape:
        prev_shape = df_med.shape
        df_movies_cnt = pd.DataFrame(df_med.groupby('movie_id').size(), columns=['count'])
        popular_movies = list(set(df_movies_cnt.query('count >= @popularity_thres').index))
        df_ratings_drop_movies = df_med[df_med.movie_id.isin(popular_movies)]
        df_users_cnt = pd.DataFrame(df_ratings_drop_movies.groupby('critic_id').size(), columns=['count'])
        prolific_users = list(set(df_users_cnt.query('count >= @popularity_thres').index))
        df_med = df_ratings_drop_movies[df_ratings_drop_movies.critic_id.isin(prolific_users)]
        new_shape = df_med.shape

    df_new = df_med
    print('\tinitial shape:', df_reviews.shape)
    print('\tnew shape:', df_new.shape)
    print('\tdistinct movies:', df_new['movie_id'].nunique())
    print('\tdistinct critics:', df_new['critic_id'].nunique())

    # this is a test user to check validity of the model
    me = pd.DataFrame([{'movie_id': 'iron_man', 'critic_id': 'test_user', 'score': 5},
                       {'movie_id': 'captain_america_civil_war', 'critic_id': 'test_user', 'score': 5},
                       {'movie_id': 'captain_america_the_winter_soldier_2014', 'critic_id': 'test_user', 'score': 5},
                       {'movie_id': 'captain_america_the_first_avenger', 'critic_id': 'test_user', 'score': 5},
                       {'movie_id': 'avengers_age_of_ultron', 'critic_id': 'test_user', 'score': 5},
                       {'movie_id': 'avengers_infinity_war', 'critic_id': 'test_user', 'score': 5},
                       {'movie_id': 'avengers_endgame', 'critic_id': 'test_user', 'score': 5}])
    df_new = df_new.append(me).reset_index(drop=True)

    print('Data loaded\n')

    model.fit(df_new)

    critic = 'test_user'
    recommendations = enumerate(model.recommend(critic, args.amount_recommendations))
    print(f'\nHere are top-{args.amount_recommendations} recommendations for critic {critic}:')
    for ind, tple in recommendations:
        print(f'\tMovie {ind+1}: {tple[0]} w score {tple[1]}')
