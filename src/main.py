
import pandas as pd
import numpy as np

from logger import logger
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

        logger.info(f"Comp matrix shape : {self.comp_matrix.shape}")
        logger.info(f"Weights number : {len(self.weights)}")

    def dijkstra_max_path(self, s):
        # в массиве дист хранить списки путей, по которым дошли в вершину
        dist = np.array(np.ones((1, self.hyp_num)) * -np.inf).ravel()

        used = np.array(np.zeros((1, self.hyp_num))).ravel()
        print(self.weights)
        dist[s] = self.weights[s]
        for i in range(self.hyp_num):
            print(dist)
            v = None

            # ищем неотмеченную вершину с максимальным расстоянием до нее
            for j in range(self.hyp_num):
                if not used[j] and (v is None or dist[j] > dist[v]):
                    v = j

            if v == None or dist[v] == -np.inf:
                break
            # отметили вершину посещенной
            used[v] = 1
            # идем по строке в матрице совместности
            for e in range(self.hyp_num):
                if e>v:
                    if self.comp_matrix[v, e] == 1:
                        if dist[v] + self.weights[e] > dist[e]:
                            print(v, e, dist[v], dist[e], self.weights[e])

                            dist[e] = dist[v] + self.weights[e]

        return dist

    def __call__(self):
        for i in range(self.hyp_num):
            dist = self.dijkstra_max_path(i)
            print(dist)
            break




def main():
    input_matrix_path = "data/input_matrix2.csv"
    out_path = "data/out.csv"
    weights_path = "data/weights2.csv"
    comp_matrix_finder = CompatabilityMatrixFinder(input_matrix_path,weights_path, out_path)
    comp_matrix_finder()
if __name__ == "__main__":
    main()