from src.use_cases.data import Data
from sklearn.datasets import load_digits


class DigitsDataset(Data):
    def __init__(self):
        print("Loading digits dataset!!!")
        digits_data = load_digits()

        feature_vector = digits_data.images.reshape(1797, 64)
        target_vector = digits_data.target

        feature_names = map(lambda index: str((index / 8, index % 8)), range(64))

        Data.__init__(self, feature_vector, target_vector, feature_names)
        print("\tDone!!")

