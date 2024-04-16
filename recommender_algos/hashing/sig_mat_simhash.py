import numpy as np
from pathlib import Path
import argparse
import math
import pandas as pd
from collections import defaultdict
import json

def generate_sig_matrix(sig_nrows,sig_ncols, r_matrix):
    r_nrows, r_ncols = r_matrix.shape
    sig_nrows = r_nrows if sig_nrows == -1 else sig_nrows
    sig_ncols = r_ncols if sig_ncols == -1 else sig_ncols

    projection = np.random.randn(sig_nrows, r_nrows)

    sig_matrix = (np.dot(projection, r_matrix) >= 0).astype(np.int32)
    return sig_matrix

def generate_hash(sig_matrix):
    sig_matrix = sig_matrix.T

    hash_values = []
    for row in range(sig_matrix.shape[0]):
        hash_values.append("".join(sig_matrix[row, :].astype("str")))

    hash_table = defaultdict(list)
    for i, h_ in enumerate(hash_values):
        hash_table[h_].append(i)

    hash_vals = defaultdict(str)
    for i, h_ in enumerate(hash_values):
        hash_vals[i] = h_
    
    return hash_table, hash_vals

def main(args):
    r_matrix = pd.read_csv(args.inputfile, header = None, index_col = None).to_numpy()

    sig_matrix = generate_sig_matrix(args.nrows, r_matrix.shape[1], r_matrix)
    df = pd.DataFrame(data = sig_matrix, index = None, columns = None)
    df.to_csv(args.outputfile, sep=",", header = None, index = None)

    if args.hashfile is not None:
        hash_table, hash_vals = generate_hash(sig_matrix)

        with open(args.hashfile, "w") as f:
            table = json.dumps(hash_table)
            values = json.dumps(hash_vals)
            f.writelines([table,"\n", values])

    with open("./.env", "a") as f:
        f.write(f'SIG_FILE="{args.outputfile}"\n')
        f.write(f'HASH_FILE="{args.hashfile}"\n')




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("nrows", type=int, help="Specify the number of rows of the signature matrix.")
    parser.add_argument("inputfile", type=str, help="Specify the complete CSV/txt file path of input matrix.")
    parser.add_argument("outputfile", type=str, help="Specify the complete CSV/txt file path to output signature matrix.")
    parser.add_argument("-hf", "--hashfile", type=str, help="Specify the complete .json file path to output hash values.")
    args = parser.parse_args()
    main(args)

    