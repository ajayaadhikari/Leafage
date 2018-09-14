from data import Data
import pandas as pd
import numpy as np


class Abalone(Data):
    def __init__(self):
        # Read data from file
        df = pd.read_csv("../data/abalone/abalone.data", header=None)

        # Add all columns as feature vector expect the age and sex
        feature_vector = df.iloc[:, 1:-1].values

        # Make the sale price discrete as ["young","old"]
        target_vector = self.split(df.iloc[:, -1].values)

        # Set the column names as the feature names
        feature_names = ["length", "diameter", "height", "whole weight",
                         "shucked weight", "viscera weight", "shell weight"]

        Data.__init__(self, feature_vector, target_vector, feature_names, name="Abalone")

    @staticmethod
    def split(ages):
        result = []
        threshold = 10

        for age in ages:
            if age < threshold:
                result.append("young")
            else:
                result.append("old")
        return np.array(result, dtype=object)