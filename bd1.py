import sys

import h5py
import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score

from loader import load_data
import argparse

parser = argparse.ArgumentParser()
# parser.add_argument("n", help="repeat time", type=int)
parser.add_argument("bd_model", help="bad model")
parser.add_argument("-n", help="treshold", type=float, dest="treshold")
parser.add_argument("-v", "--validate", dest="validate_data_path")
parser.add_argument("-t", "--test", dest="test_data_path")
parser.add_argument("-p", "--poisoned", dest="poisoned_data_path")
parser.add_argument("-m", "--mode", dest="mode", default="repair")

args = parser.parse_args()

BD_model = keras.models.load_model(args.bd_model)
N = 1283
dim_n = 55*47*3
VECTOR_VALI_FILE = "./data/validate_vector.txt"

def load_vector(x):
    feat1 = keras.Model(inputs=BD_model.input,
                    outputs=BD_model.get_layer('flatten_1').output)
    feat2 = keras.Model(inputs=BD_model.input,
                    outputs=BD_model.get_layer('flatten_2').output)
    return np.concatenate([feat1(x),feat2(x)], axis=1)

def is_backdoor_class(data, label=0):
    return True if BD_model.predict(data)==label else False

def main():

    x_test, y_test = load_data(args.test_data_path)
    vector_test = load_vector(x_test[y_test.astype(int)==0])

    if args.mode == 'repair':
        x_veli, y_veli = load_data(args.validate_data_path)
        vector_validation = load_vector(x_veli[y_veli.astype(int)==0])
        mutual_info = []
        for i in range(9):
            mutual_info.append(normalized_mutual_info_score(vector_test[i], np.mean(vector_validation, axis=0)))
        print("range between: ", min(mutual_info), max(mutual_info))
        np.savetxt(VECTOR_VALI_FILE, np.mean(vector_validation, axis=0), delimiter=',')
        
    elif args.mode == 'evaluate':
        treshold = args.treshold
        count_test, count_poisoned = 0, 0
        x_poisoned, y_poisoned = load_data(args.poisoned_data_path)
        vector_poisoned = load_vector(x_poisoned)
        validate_vector = np.loadtxt(VECTOR_VALI_FILE, delimiter=',')
        for x in vector_test:
            score = normalized_mutual_info_score(x, validate_vector)
            if score > treshold:
                count_test += 1
        for x in vector_poisoned:
            score = normalized_mutual_info_score(x, validate_vector)
            if score < treshold:
                count_poisoned += 1
        print("filter test accuracy: {}/{}, poisoned accuracy: {}/ {}".format(count_test, len(vector_test), count_poisoned, len(vector_poisoned)))
    
    
if __name__ == '__main__':
    main()
