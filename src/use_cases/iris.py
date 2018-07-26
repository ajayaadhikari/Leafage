from src.use_cases.data import Data
from sklearn.datasets import load_iris


class IrisDataSet(Data):
    def __init__(self):
        iris = load_iris()

        feature_vector = iris.data
        target_vector = iris.target

        feature_names = iris.feature_names
        class_names = iris.target_names

        categorical_features = None
        categorical_names = None

        Data.__init__(self, feature_vector, target_vector, feature_names, class_names, categorical_features, categorical_names)