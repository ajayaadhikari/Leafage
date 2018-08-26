from data import Data
from sklearn.datasets import load_breast_cancer


class BreastCancerDataset(Data):
    def __init__(self):
        cancer = load_breast_cancer()

        feature_vector = cancer.data
        target_vector = [cancer.target_names[i] for i in cancer.target]

        feature_names = cancer.feature_names

        Data.__init__(self, feature_vector, target_vector, feature_names, name="BreastCancer")
