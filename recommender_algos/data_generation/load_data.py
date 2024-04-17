import os
import sys
sys.path.append(os.path.join(os.getcwd(), "recommender_algos"))
from data_generation.get_data import read_params, get_data

import argparse

def load_and_save(config_path):
    config = read_params(config_path)
    df = get_data(config_path)
    df = df.pivot(index = "User_ID", columns = "Movie_ID", values = "Rating")
    df.to_csv(config["load_data"]["dataset_csv"])

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    load_and_save(config_path=parsed_args.config)    