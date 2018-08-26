from data import Data
from sklearn.datasets import load_wine


class WineDataset(Data):
    def __init__(self):
        wine = load_wine()

        feature_vector = wine.data
        target_vector = [wine.target_names[i] for i in wine.target]

        feature_names = wine.feature_names

        Data.__init__(self, feature_vector, target_vector, feature_names, name="Wine")
