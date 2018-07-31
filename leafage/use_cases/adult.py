from sklearn.model_selection import train_test_split

from data import Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np


class Adult(Data):
    def __init__(self):
        data_set, target_vector, class_names = Adult.get_dataset()
        data_set, target_vector = self.reduce_size(0.005, data_set, target_vector)

        feature_names = ["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital Status", "Occupation",
                         "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss", "Hours per week", "Country"]

        categorical_features = [1, 3, 5, 6, 7, 8, 9, 13]
        categorical_names = {}
        for feature in categorical_features:
            le = LabelEncoder()
            le.fit(data_set[:, feature])
            data_set[:, feature] = le.transform(data_set[:, feature])
            categorical_names[feature] = le.classes_

        data_set = data_set.astype(float)
        encoder = OneHotEncoder(categorical_features=categorical_features)
        encoder.fit(data_set)

        Data.__init__(self, data_set, target_vector, feature_names, class_names, encoder)

    @staticmethod
    def reduce_size(ratio, dataset, target_vector):
        train, _, labels_train, _ = train_test_split(dataset,
                                                     target_vector,
                                                     train_size=ratio,
                                                     stratify=target_vector)
        return train, labels_train

    @staticmethod
    def get_dataset():
        data = np.genfromtxt('../data/adult/adult.data', delimiter=', ', dtype=str)
        labels = data[:, 14]
        le = LabelEncoder()
        le.fit(labels)
        target_labels = le.transform(labels)
        class_names = le.classes_
        return data[:, :-1], target_labels, class_names
