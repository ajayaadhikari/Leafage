from data import Data, PreProcess
import pandas as pd


class FileDataSet(Data):
    def __init__(self, path):
        # Read data from file
        df = pd.read_csv(path)

        # Add all columns as feature vector expect the sale price
        feature_vector = df.iloc[:, 0:-1].values

        # Make the sale price discrete as ["low", "medium", "high"]
        target_vector = df.iloc[:, -1].values

        # Set the column names as the feature names
        feature_names = list(df)[:-1]

        Data.__init__(self, feature_vector, target_vector, feature_names)
