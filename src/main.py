import itertools

import pandas as pd
import numpy as np
from time import time


class CompatabilityMatrixFinder:
    def __init__(self, input_matrix_path, weights_path, out_path):
        self.weights = None
        self.input_df = None
        self.out_df = None
        self.hyp_num = None

        self.__read_data(input_matrix_path, weights_path, out_path)

    def __read_data(self, input_matrix_path, weights_path, out_path = None):
        weights = pd.read_csv(weights_path, header=None).to_numpy()
        self.weights = weights.reshape(-1)

        input_df = pd.read_csv(input_matrix_path, header=None).to_numpy()
        self.comp_matrix = np.maximum(input_df, input_df.transpose())

        if out_path is not None:
            self.out_df = pd.read_csv(out_path, header=None).to_numpy()

        self.hyp_num = len(self.weights)


    def is_compatable(self, path, e):
        # print(path, e)
        comp_matrix_e = self.comp_matrix[e, :]
        comp_values = [comp_matrix_e[i] for i in path]
        return comp_values == [1]*len(comp_values)

    def delete_duplicates(self, list_paths_lengths):
        pass
    def dijkstra_max_path(self, s):
        # в массиве дист хранить списки путей, по которым дошли в вершину и их длины
        # dist[i] = [ (path1 = [1,2,3,5], len = 7), (path2, len)   ]
        dist = [[] for i in range(self.hyp_num)]

        dist[s] = [[[s], self.weights[s]]]
        for i in range(self.hyp_num):
            v = i
            # идем по строке в матрице совместности

            for e in range(v+1, self.hyp_num):
                if self.comp_matrix[e, v] == 1:
                    for path_i in range(len(dist[v])):
                        path_length_cur = dist[v][path_i]
                        # print(dist, dist[v], path_length_cur)
                        path = path_length_cur[0]
                        length = path_length_cur[1]

                        if self.is_compatable(path, e):
                            dist[e].append([path + [e], length + self.weights[e]])

                        # dist[e] = self.delete_duplicates(dist[e])

        print(s, dist, '\n')
        return dist

    def __call__(self):
        max_len = 0
        path_max_len = []

        st = time()
        dist = self.dijkstra_max_path(0)
        print(f"Dijkstra run time {time()-st}")
        dist = list(itertools.chain(*dist))

        # дописать что массив dist в конце
        # после сортируется по 2 элементу и выводятся топ - х
        # путей с их весами

        # код ниже нужен временно чтобы мы понимали верно ли работает вообще
        for el in dist:
            cur_path_len = el[1]
            cur_path = el[0]

            if cur_path_len > max_len:
                max_len = cur_path_len
                path_max_len = cur_path

        print(max_len, path_max_len)


def main():
    input_matrix_path = "data/input_matrix.csv"
    out_path = "data/out.csv"
    weights_path = "data/weights.csv"
    comp_matrix_finder = CompatabilityMatrixFinder(input_matrix_path,weights_path, out_path)
    comp_matrix_finder()


if __name__ == "__main__":
    main()