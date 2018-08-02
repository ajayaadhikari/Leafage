from data import Data, PreProcess
from sklearn.datasets import load_iris


class IrisDataSet(Data):
    def __init__2(self):
        iris = load_iris()

        feature_vector = iris.data
        target_vector = [iris.target_names[i] for i in iris.target]

        feature_names = iris.feature_names

        Data.__init__(self, feature_vector, target_vector, feature_names)

    def __init__(self):
        import numpy as np
        iris = load_iris()

        feature_vector = iris.data.tolist()
        new_column1 = (len(feature_vector)/2)*["a"] + (len(feature_vector)/2)*["b"]
        new_column2 = (len(feature_vector)/2)*["c"] + (len(feature_vector)/2)*["d"]

        feature_vector = [[new_column2[i]] + feature_vector[i] + [new_column1[i]] for i in range(len(feature_vector))]
        feature_vector = np.array(feature_vector, dtype=object)

        target_vector = [iris.target_names[i] for i in iris.target]

        feature_names = iris.feature_names
        feature_names.append("test")
        feature_names.append("test2")

        Data.__init__(self, feature_vector, target_vector, feature_names)