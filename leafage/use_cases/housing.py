from data import Data, PreProcess
import pandas as pd
import numpy as np


class HousingDataSet(Data):
    def __init__(self):
        # Read data from file
        df = pd.read_csv("../data/housing/train.csv", na_values="NA")
        #df.dropna(how="any", inplace=True, axis=1)
        #df = df.iloc[:,1:]
        #df["SalePrice"] = self.split(df.iloc[:, -1].values)
        #df.to_csv("../data/housing/pre_processed_train.csv",index=False)
#
        #df2 = pd.read_csv("../data/housing/test.csv", na_values="NA")
        #df2.dropna(how="any", inplace=True, axis=1)
        #df2 = df2.iloc[:,1:-1]
        #df2.to_csv("../data/housing/pre_processed_test.csv", index=False)

        # Drop columns with at least one null value
        df.dropna(how="any", inplace=True, axis=1)

        # Add all columns as feature vector expect the sale price
        feature_vector = df.iloc[:, 1:-1].values

        # Make the sale price discrete as ["low", "medium", "high"]
        target_vector = self.split(df.iloc[:, -1].values)

        # Set the column names as the feature names
        feature_names = list(df)[1:-1]

        Data.__init__(self, feature_vector, target_vector, feature_names)

    @staticmethod
    def split(sale_price):
        result = []
        first_threshold = 150000
        second_threshold = 200000

        for price in sale_price:
            if price <= first_threshold:
                result.append("low")
            elif price > first_threshold and price <= second_threshold:
                result.append("medium")
            else:
                result.append("high")
        return np.array(result, dtype=object)
