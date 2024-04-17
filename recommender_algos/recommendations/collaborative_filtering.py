import numpy as np
import argparse
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from recommender_algos.data_generation.get_data import read_params
# import os
# import sys
# sys.path.append(os.path.join( os.getcwd(), "recommender_algos"))
# from data_generation.get_data import read_params

def recommendation_classic(r_matrix, itemId, userId):
    r_avg = (np.sum(r_matrix, axis = 1) / np.sum( np.array(r_matrix > 0, dtype = np.int32) , axis = 1)) [:, np.newaxis]
    inner_prods = np.sum( np.array(r_matrix > 0, dtype = np.int32) * (r_matrix - r_avg)  *  (r_matrix[:, itemId:itemId+1] - r_avg), axis = 0)

    sigs = np.sqrt(np.sum(np.square(r_matrix), axis = 0))
    sim_den_term = sigs * sigs[itemId]

    sims = inner_prods / sim_den_term
    sims_sum = np.sum(sims)
    
    item_means = np.sum(r_matrix, axis = 0) / np.sum( np.array( r_matrix > 0, dtype = np.int32) , axis = 0)
    prods_sum = np.sum ( sims * (r_matrix[userId, :] -  item_means))

    return round(item_means[itemId] + (prods_sum / sims_sum))

def recommendation_knn(r_matrix, itemId, userId):
    knn = NearestNeighbors(n_neighbors = 5, metric = "cosine")
    knn.fit(r_matrix.T)

    distances, indices = knn.kneighbors(r_matrix.T[itemId:itemId+1, :], n_neighbors = 5)
    # print(distances, indices)

    return indices.flatten()[-1]

def main(args):
    config = read_params(args.config)
    df = pd.read_csv(config["load_data"]["dataset_csv"], index_col= "User_ID").fillna(0)
    r_matrix = df.to_numpy()[1:, 1:]

    itemId, userId = config["recommend"]["itemId"], config["recommend"]["userId"]
    normal = config["recommend"]["normal"]
    if normal:
        print(recommendation_classic(r_matrix, itemId, userId))
    else:
        print(recommendation_knn(r_matrix, itemId, userId))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    main(args)