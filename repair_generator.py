import argparse
import sys

import h5py
import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score

from loader import load_data

N_class = 1283
parser = argparse.ArgumentParser()
parser.add_argument("bd_model", help="bad model")
parser.add_argument("-v", "--validate", dest="validate_data_path")
parser.add_argument("-t", "--test", dest="test_data_path")
parser.add_argument("-m", "--mode", dest="mode", default="pca")

args = parser.parse_args()


class BD():

    def __init__(self, bd):
        self.posioned_labels = None
        self.bd = bd
        self.treshold = {}
        self.mean_vector = {}
        self.pca = {}

    def load_vector(self, x):
        feat1 = keras.Model(inputs=self.bd.input,
                            outputs=self.bd.get_layer('flatten_1').output)
        feat2 = keras.Model(inputs=self.bd.input,
                            outputs=self.bd.get_layer('flatten_2').output)
        return np.concatenate([feat1(x), feat2(x)], axis=1)

    def get_treshold(self, vec_vali_x, vec_vali_y):
        for i in range(N_class):
            idx = (vec_vali_y == i)
            vec_x = vec_vali_x[idx]
            mean_vec_x = np.mean(vec_x[:len(vec_x) // 2], axis=0)
            score = float('inf')
            for j in range(len(vec_x) // 2, len(vec_x)):
                score_ = normalized_mutual_info_score(
                    vec_vali_x[j], mean_vec_x)
                if score_ < score:
                    score = score_
            self.treshold[i] = score
            self.mean_vector[i] = mean_vec_x

    def get_pca_treshold(self, vector_x, y):
        for i in range(N_class):
            idx = (y == i)
            matrix = np.dot(
                self.mean_vector.get(i),
                np.transpose(
                    vector_x[idx]))
            self.treshold[i] = np.linalg.norm(matrix, axis=0, keepdims=True)

    def fit_pca(self, vector_x, y=None):
        pca = PCA(n_components=2)
        for i in range(N_class):
            idx = (y == i)
            x = pca.fit_transform(np.transpose(vector_x[idx]))
            self.mean_vector[i] = np.transpose(x)

    def pca_filter(self, xs, ys):
        result = [False] * len(ys)
        for i, (x, y) in enumerate(zip(xs, ys)):
            temp = np.dot(self.mean_vector.get(y), np.transpose(x))
            if np.linalg.norm(temp) < np.min(
                    self.treshold[y]) or np.linalg.norm(temp) > np.max(
                    self.treshold[y]):
                result[i] = True
        return result

    def filter(self, xs, ys):
        result = [False] * len(ys)
        for i, (x, y) in enumerate(zip(xs, ys)):
            if normalized_mutual_info_score(
                    self.mean_vector.get(y),
                    x) < self.treshold.get(y):
                result[i] = True
        return result

    def get_result(self, y_predict, idx):
        label = np.array(y_predict)
        label[idx] = N_class
        return label


def main():
    original_bd = keras.models.load_model(args.bd_model)
    bd = BD(original_bd)
    # Load data
    x_vali, y_vali = load_data(args.validate_data_path)

    if args.mode == "pca":
        # First Step -> get treshold
        vector_validation_x = bd.load_vector(x_vali)
        bd.fit_pca(vector_validation_x, y_vali)
        bd.get_pca_treshold(vector_validation_x, y_vali)

        # Second Step -> filter poisoned
        x_test, _ = load_data(args.test_data_path)
        y_test_predict = np.argmax(
            original_bd.predict(x_test), axis=1).reshape(-1)
        vector_test_x = bd.load_vector(x_test)
        idx = bd.pca_filter(vector_test_x, y_test_predict)

        print(sum(idx), len(y_test_predict))
        # Third Step -> return correct label
        label = bd.get_result(y_test_predict, idx)
        print("{} / {} / {}".format(sum(idx), sum(label != 5), len(x_test)))
        print(label.tolist())

    else:
        # First Step -> get treshold
        vector_validation_x = bd.load_vector(x_vali)
        bd.get_treshold(vector_validation_x, y_vali)

        # Second Step -> filter poisoned
        x_test, y_test = load_data(args.test_data_path)
        y_test_predict = np.argmax(
            original_bd.predict(x_test), axis=1).reshape(-1)
        vector_test_x = bd.load_vector(x_test)
        idx = bd.filter(vector_test_x, y_test_predict.tolist())

        # Third Step -> return correct label
        label = bd.get_result(y_test_predict, idx)
        print(label.tolist())
        # print(sum(idx), " ", len(label))



if __name__ == '__main__':
    main()
