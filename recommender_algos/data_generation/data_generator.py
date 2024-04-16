import numpy as np
from pathlib import Path
import argparse
import math
import pandas as pd

def generate_matrix(nrows, ncols, binary = False, threshold = 0.7, sparsity_level = 0.7):
    res_matrix = None
    if binary:
        for i in range(ncols):
            col = np.random.binomial(n = 1, p = threshold, size = (nrows, 1))
            if res_matrix is None: res_matrix = col
            else: res_matrix = np.concatenate([res_matrix, col], axis = 1)
    else:
        for i in range(ncols):
            col = np.ones(shape = (nrows, 1), dtype = np.int32)

            vec = np.round(np.random.normal(loc = 2.5, scale = 1.0, size = (math.floor(nrows * sparsity_level), 1)), decimals = 0).astype(np.int32)
            idxs = np.random.choice(list(range(23)), size = math.floor(nrows * sparsity_level))
            col[idxs] = vec

            col[col == 0] = 1

            if res_matrix is None: res_matrix = col
            else: res_matrix = np.concatenate([res_matrix, col], axis = 1)
    return res_matrix

def main(args):
    if args.dimensions is not None:
        nrows, ncols = list(map(int, args.dimensions.split("*")))

        r_matrix = generate_matrix(nrows, ncols, binary = args.binary,
                                    threshold = args.threshold if args.threshold is not None else 0.7,
                                    sparsity_level = args.sparsity if args.sparsity is not None else 0.7)

        if args.file is not None:
            df = pd.DataFrame(data = r_matrix, index = None, columns = None)
            df.to_csv(args.file, sep=",", header = None, index = None)

    with open("./.env", "a") as f:
        f.write(f'R_FILE="{args.file}"\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dim", "--dimensions", type=str, help="Specify the dimension of the ratings matrix number_of_rows*number_of_cols format")
    parser.add_argument("-b", "--binary", action="store_true", help="Specify whether the ratings are binary, default is non_binary")
    parser.add_argument("-f", "--file", type=str, help="Specify the conplete CSV file path to store the matrix.")
    parser.add_argument("-t", "--threshold", type=float, help="Specify the threshold level in case binary=True. default is 0.7")
    parser.add_argument("-s", "--sparsity", type=float, help="Specify the sparsity level in case binary=False. default is 0.7")
    args = parser.parse_args()
    main(args)
    