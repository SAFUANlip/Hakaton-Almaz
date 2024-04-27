from time import time

import networkx as nx
import numpy as np
import pandas as pd

from custom_lib.custom_clique_alg import custom_max_weight_clique


class CompatabilityMatrixFinder:
    def __init__(self, input_matrix_path, weights_path, out_path, pred_path):
        self.weights = None
        self.input_df = None
        self.out_df = None
        self.hyp_num = None
        self.pred_path = pred_path

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
        self.multiplic = 10 ** 5
        G = nx.Graph()

        for i in range(len(self.weights)):
            G.add_node(i, weight=int(self.weights[i] * self.multiplic))

        for i in range(len(self.weights)):
            for j in range(i + 1, len(self.weights)):
                if self.comp_matrix[i, j] == 1:
                    G.add_edge(i, j)

        output = custom_max_weight_clique(G)
        return output[2]

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

    def cvt_path2gh(self, path):
        ans = np.zeros((1, self.hyp_num), np.uint8).ravel()
        for el in path:
            ans[el] = int(1)

        return ans

    def create_pred_df(self, top5_dict):
        hyp_names = ["TH"+ str(i+1) for i in range(self.hyp_num)]
        pred_df = pd.DataFrame(columns=hyp_names + ["sum(w)"])

        for key in sorted(top5_dict.keys(), reverse=False):
            cur_path = top5_dict[key]
            cur_weight = key / self.multiplic
            cur_path_gh = self.cvt_path2gh(cur_path)

            pred_df.loc[-1] = list(cur_path_gh) + [cur_weight]  # adding a row
            pred_df.index = pred_df.index + 1  # shifting index
            pred_df = pred_df.sort_index()

        pred_df[hyp_names] = pred_df[hyp_names].astype(int)
        pred_df.to_csv(self.pred_path, index=False)

    def count_gh_weight(self):
        for i in range(1,self.out_df.shape[0] ):
            counted_ans = 0
            cur_row = self.out_df[i,:]
            cur_path = cur_row[1 :-1]
            cur_ans = float(cur_row[-1])

            for j in range(len(cur_path)):
                if "1" in cur_path[j]:
                    counted_ans += self.weights[j]

            print(cur_ans, counted_ans)

    def __call__(self):
        st = time()

        top5_dict = self.max_clique_solution()
        top5_dict = dict(sorted(top5_dict.items(), reverse=True))
        print(f"Поиск весов занял {time() - st} секунд, ответ лежит в {self.pred_path}")
        st = time()

        self.create_pred_df(top5_dict)
        print(f"df создался за {time() - st} секунд")


def main():
    st = time()
    input_matrix_path = "data/input_matrix1.csv"
    out_path = "data/out1.csv"
    weights_path = "data/weights1.csv"
    pred_path = "data/pred1.csv"
    comp_matrix_finder = CompatabilityMatrixFinder(input_matrix_path, weights_path, out_path, pred_path)
    # comp_matrix_finder.count_gh_weight()
    comp_matrix_finder()

    print(f"Решение отработало за {time()-st} секунд")


if __name__ == "__main__":
    main()
