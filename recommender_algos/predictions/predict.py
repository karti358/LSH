import argparse
import numpy as np
import pandas as pd

from recommender_algos.recommendations import *
from recommender_algos.data_generation.get_data import read_params

def load_matrix():
    config = read_params(config_path="params.yaml")
    df = pd.read_csv(config["load_data"]["dataset_csv"], index_col = "User_ID").fillna(0)
    r_matrix = df.to_numpy().astype(np.int32)
    print(r_matrix)
    return r_matrix

def predict(r_matrix, itemId, userId, normal):
    if normal:
        return recommendation_classic(r_matrix, itemId, userId)
    return recommendation_knn(r_matrix, itemId, userId)

def form_response(r_matrix, request_form):
    userId, itemId = list(map(int, request_form.values()))
    if r_matrix[userId, itemId] != 0:
        return r_matrix[userId, itemId]
    return predict(r_matrix, itemId, userId, True)

def api_response(r_matrix, request_json):
    userId, itemId  = list(map(int, request_json.values()))
    if r_matrix[userId, itemId] != 0:
        return r_matrix[userId, itemId]
    return predict(r_matrix, itemId, userId, True)

def main(config):
    df = pd.read_csv(config["load_data"]["dataset_csv"], index_col= "User_ID").fillna(0)
    r_matrix = df.to_numpy().astype(np.int32)
    return predict(r_matrix, config["recommend"]["itemId"], config["recommend"]["userId"], config["recommend"]["normal"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    config = read_params(args.config)
    print(main(config))
    