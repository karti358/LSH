import numpy as np
import argparse
import pandas as pd
from sklearn.neighbors import NearestNeighbors

def recommendation_classic(r_matrix, itemId, userId):
    r_avg = (np.sum(r_matrix, axis = 1) / np.sum( np.array(r_matrix > 0, dtype = np.int32) , axis = 1)) [:, np.newaxis]
    inner_prods = np.sum( np.array(r_matrix > 0, dtype = np.int32) * (r_matrix - r_avg)  *  (r_matrix[:, itemId:itemId+1] - r_avg), axis = 0)

    sigs = np.sqrt(np.sum(np.square(r_matrix), axis = 0))
    sim_den_term = sigs * sigs[itemId]

    sims = inner_prods / sim_den_term
    sims_sum = np.sum(sims)
    
    item_means = np.sum(r_matrix, axis = 0) / np.sum( np.array( r_matrix > 0, dtype = np.int32) , axis = 0)
    prods_sum = np.sum ( sims * (r_matrix[userId, :] -  item_means))

    # print("prods_sum: ", prods_sum)
    # print("sims_sum: ", sims_sum)
    # print("division -", prods_sum / sims_sum)
    # print("item_means: ", item_means, item_means[itemId-1:itemId+2])
    # print(np.sum(r_matrix, axis = 0)[itemId])
    # print(np.sum( np.array( r_matrix > 0, dtype = np.int32 ) , axis = 0)[itemId])

    return round(item_means[itemId] + (prods_sum / sims_sum))

def recommendation_knn(r_matrix, itemId, userId):
    knn = NearestNeighbors(n_neighbors = 5, metric = "cosine")
    knn.fit(r_matrix.T)

    distances, indices = knn.kneighbors(r_matrix.T[itemId:itemId+1, :], n_neighbors = 5)
    # print(distances, indices)

    return indices.flatten()[-1]



def main(args):
    df = pd.read_csv(args.inputfile, index_col= "User_ID", header = 0).fillna(0)
    r_matrix = df.to_numpy()

    if args.normal:
        print(recommendation_classic(r_matrix, args.itemId, args.userId))
    else:
        print(recommendation_knn(r_matrix, args.itemId, args.userId))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("userId", type=int, help="Specify the user ID for which recommending.")
    parser.add_argument("itemId", type=int, help="Specify the item ID being recommended.")
    parser.add_argument("inputfile", type=str, help="Specify the complete CSV/txt file path to input ratings matrix.") 
    parser.add_argument('--normal', action=argparse.BooleanOptionalAction, help = "Normal recommendation or Knn type")
    
    args = parser.parse_args()
    main(args)