from sklearn.model_selection import train_test_split

from data import Data
import pandas as pd


class Adult(Data):
    def __init__(self):
        feature_names = ["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital Status", "Occupation",
                         "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss", "Hours per week", "Country"]

        data = pd.read_csv('../data/adult/adult.data', na_values="NA", names=feature_names+["Salary"], header=None).values
        labels = data[:, 14]

        Data.__init__(self, data[:, :-1], labels, feature_names, name="Adult")

    @staticmethod
    def reduce_size(ratio, dataset, target_vector):
        train, _, labels_train, _ = train_test_split(dataset,
                                                     target_vector,
                                                     train_size=ratio,
                                                     stratify=target_vector)
        return train, labels_train

