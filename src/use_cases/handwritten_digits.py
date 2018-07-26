from src.use_cases.data import Data
from sklearn.datasets import load_digits


class Digits(Data):
    def __init__(self):
        print("Loading digits dataset!!!")
        digits_data = load_digits()

        original_feature_vector = digits_data.images.reshape(1797, 64)
        # binary_feature_vector = np.vectorize(lambda x: 0 if x == 0 else 1)(original_feature_vector)
        target_vector = digits_data.target

        feature_names = map(lambda index: str((index / 8, index % 8)), range(64))
        class_names = digits_data.target_names

        categorical_features = None
        categorical_names = None

        Data.__init__(self, original_feature_vector, target_vector, feature_names, class_names, categorical_features,
                    categorical_names)
        print("\tDone!!")

