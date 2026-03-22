from abc import ABC, abstractmethod
import numpy as np


class BaseInfluenceMethod(ABC):
    @abstractmethod
    def compute_score(self, test_info: dict, train_info: dict) -> float:
        ...


class InfluenceCalculator:
    def __init__(self, method: BaseInfluenceMethod):
        self.method = method

    def compute_matrix(self, test_infos: list, train_infos: list) -> np.ndarray:
        n_test = len(test_infos)
        n_train = len(train_infos)
        matrix = np.zeros((n_test, n_train))

        for i, t_info in enumerate(test_infos):
            for j, tr_info in enumerate(train_infos):
                matrix[i, j] = self.method.compute_score(t_info, tr_info)

        return matrix
