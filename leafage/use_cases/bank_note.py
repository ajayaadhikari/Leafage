from data import Data
import pandas as pd


class BankNote(Data):
    def __init__(self):
        # Read data from file
        df = pd.read_csv("../data/bank_note/data_banknote_authentication.csv", header=None)

        # Add all columns as feature vector expect the target column
        feature_vector = df.iloc[:, :-1].values

        target_vector = df.iloc[:, -1].values

        # Set the column names as the feature names
        feature_names = ["variance of WTI", "skewness of WTI", "curtosis of Wavelet", "entropy of image"]

        Data.__init__(self, feature_vector, target_vector, feature_names, name="BankNote")