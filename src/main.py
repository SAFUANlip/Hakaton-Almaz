import itertools
import numpy as np
from custom_lib.custom_clique_alg import custom_max_weight_clique


class CompatabilityMatrixFinder:
    def __init__(self, pred_path, input_with_weights_path):
        self.weights = None
        self.input_df = None
        self.out_df = None
        self.hyp_num = None
        self.pred_path = pred_path

        self.__read_data(input_with_weights_path)

    def __read_data(self, input_with_weights_path):
        input_with_weights = np.loadtxt(input_with_weights_path, delimiter=',')

        self.weights = input_with_weights[-1, :]
        self.comp_matrix = input_with_weights[:-1, :]

        self.hyp_num = len(self.weights)

    def __is_compatable(self, path, e):
        comp_matrix_e = self.comp_matrix[:, e]
        comp_values = [comp_matrix_e[i] for i in path]
        return comp_values == [1] * len(comp_values)

    def max_clique_solution(self):
        self.multiplic = 10 ** 5
        output = custom_max_weight_clique(self.multiplic, self.weights, self.comp_matrix)
        return output[2]

    def dijkstra_solution(self):
        s = 0

        # в массиве дист хранить списки путей, по которым дошли в вершину и их длины
        dist = [[] for i in range(self.hyp_num)]

        dist[s] = [[[s], self.weights[s]]]
        for i in range(self.hyp_num):
            v = i
            # идем по строке в матрице совместности

            for e in range(v + 1, self.hyp_num):
                if self.comp_matrix[v, e] == 1:
                    for path_i in range(len(dist[v])):
                        path_length_cur = dist[v][path_i]
                        path = path_length_cur[0]
                        length = path_length_cur[1]

                        if self.__is_compatable(path, e):
                            dist[e].append([path + [e], length + self.weights[e]])

        dist = list(itertools.chain(*dist))
        dist.sort(key=lambda x: x[1], reverse=True)
        ans_list = dist[:5]

    def __check_is_accept(self, comp_traj, candidate, index):
        for i in range(len(candidate)):
            if candidate[i] == 1 and comp_traj[i][index] == 0:
                return False
        return True

    def __recurent_search(self, comp_traj, candidate_cur):
        if len(candidate_cur) == self.hyp_num:
            self.available_hyps.append(candidate_cur)
            return
        if self.__check_is_accept(comp_traj, candidate_cur, len(candidate_cur)):
            self.__recurent_search(comp_traj, candidate_cur + [1])
        self.__recurent_search(comp_traj, candidate_cur + [0])

    def complete_search_solution(self):
        self.available_hyps = []
        comp_matrix_symm = np.maximum(self.comp_matrix, self.comp_matrix)
        self.__recurent_search(comp_matrix_symm, [1])
        self.__recurent_search(comp_matrix_symm, [0])
        dist = []

        for hyp in self.available_hyps:
            dist.append([hyp, sum(np.array(hyp) * np.array(self.weights))])

        dist.sort(key=lambda x: x[1], reverse=True)
        ans_list = dist[:5]


    def __cvt_path2gh(self, path):
        ans = ['0,' for i in range(self.hyp_num)]
        for el in path:
            ans[el] = "1,"

        return ans

    def __create_pred_df(self, top5_dict):
        hyp_names = ["TH" + str(i + 1) for i in range(self.hyp_num)] + ["sum(w)\n"]
        hyp_names_str = ",".join(hyp_names)
        data_pred = [hyp_names_str]

        for key in sorted(top5_dict.keys(), reverse=True):
            cur_path = top5_dict[key]
            cur_weight = key / self.multiplic
            cur_path_gh = self.__cvt_path2gh(cur_path)
            ans_list = cur_path_gh + [str(cur_weight) + "\n"]
            ans_string = "".join(ans_list)
            data_pred.append(ans_string)  # adding a row

        with open(self.pred_path, "w") as file:
            for string in data_pred:
                file.write(string)

    def __call__(self):
        top5_dict = self.max_clique_solution()
        top5_dict = dict(sorted(top5_dict.items(), reverse=True))

        self.__create_pred_df(top5_dict)


def main():
    input_with_weights_path = "../data/input_with_weights.csv"
    pred_path = "../data/pred1.csv"

    comp_matrix_finder = CompatabilityMatrixFinder(pred_path, input_with_weights_path)
    comp_matrix_finder()

if __name__ == "__main__":
    main()
