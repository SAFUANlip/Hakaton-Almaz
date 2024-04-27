import itertools
from time import time

import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.clique import max_weight_clique


class CompatabilityMatrixFinder:
    def __init__(self, input_matrix_path, weights_path, out_path):
        self.weights = None
        self.input_df = None
        self.out_df = None
        self.hyp_num = None

        self.__read_data(input_matrix_path, weights_path, out_path)

    def __read_data(self, input_matrix_path, weights_path, out_path=None):
        weights = pd.read_csv(weights_path, header=None).to_numpy()
        self.weights = weights.reshape(-1)

        input_df = pd.read_csv(input_matrix_path, header=None).to_numpy()
        self.comp_matrix = input_df

        if out_path is not None:
            self.out_df = pd.read_csv(out_path, header=None).to_numpy()

        self.hyp_num = len(self.weights)

    def __is_compatable(self, path, e):
        comp_matrix_e = self.comp_matrix[e, :]
        comp_values = [comp_matrix_e[i] for i in path]
        return comp_values == [1] * len(comp_values)

    def max_clique_solution(self):
        multiplic = 10 ** 5
        G = nx.Graph()

        for i in range(len(self.weights)):
            G.add_node(i, weight=int(self.weights[i] * multiplic))

        for i in range(len(self.weights)):
            for j in range(i + 1, len(self.weights)):
                if self.comp_matrix[i, j] == 1:
                    G.add_edge(i, j)

        output = max_weight_clique(G)
        return output[0], output[1] / multiplic

    def dijkstra_solution(self, s):
        # в массиве дист хранить списки путей, по которым дошли в вершину и их длины
        dist = [[] for i in range(self.hyp_num)]

        dist[s] = [[[s], self.weights[s]]]
        for i in range(self.hyp_num):
            v = i
            # идем по строке в матрице совместности

            for e in range(v + 1, self.hyp_num):
                if self.comp_matrix[e, v] == 1:
                    for path_i in range(len(dist[v])):
                        path_length_cur = dist[v][path_i]
                        # print(dist, dist[v], path_length_cur)
                        path = path_length_cur[0]
                        length = path_length_cur[1]

                        if self.__is_compatable(path, e):
                            dist[e].append([path + [e], length + self.weights[e]])
        return dist

    def __call__(self):
        st = time()
        ans = np.zeros((1, self.hyp_num), np.uint8).ravel()

        path, weight = self.max_clique_solution()

        for el in path:
            ans[el] = 1

        path.sort()
        print(ans, path, weight)
        print(f"Решение отработало за {time() - st} секунд")


def main():
    input_matrix_path = "data/input_matrix4.csv"
    out_path = "data/out.csv"
    weights_path = "data/weights4.csv"
    comp_matrix_finder = CompatabilityMatrixFinder(input_matrix_path, weights_path, out_path)
    comp_matrix_finder()


if __name__ == "__main__":
    main()
