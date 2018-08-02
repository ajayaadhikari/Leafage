from sklearn.preprocessing import StandardScaler, LabelEncoder,OneHotEncoder
import numpy as np


class InvertibleOneHotEncoder(OneHotEncoder):
    # feature_vector should be an np.array with dtype=object if categorical features are present
    def __init__(self, X, categorical_features):
        OneHotEncoder.__init__(self, categorical_features=categorical_features, sparse=False)
        self.fit(X)

    def __inverse_transform_single(self, x):
        x_copy = x.tolist()
        start_indices = self.feature_indices_[:-1]
        end_indices = self.feature_indices_[1:]

        new_columns = []
        for start_index, end_index in zip(start_indices, end_indices):
            slice = map(int, x_copy[start_index:end_index])
            max_index = np.argmax(slice)
            new_columns.append(max_index)

        for index in range(self.feature_indices_[-1]-1, -1, -1):
            del x_copy[index]

        for new_value, index in zip(new_columns, self.categorical_features):
            x_copy.insert(index, new_value)

        return x_copy

    def inverse_transform(self, X):
        return np.array([self.__inverse_transform_single(x) for x in X])


class FeatureVectorLabelEncoder:
    # feature_vector should be an np.array with dtype=object if categorical features are present
    def __init__(self, feature_vector, categorical_features):
        self.categorical_features = categorical_features
        self.encoders = self.__get_encoders(feature_vector)

    def __get_encoders(self, feature_vector):
        label_encoders = {}

        for feature_index in self.categorical_features:
            label_encoder = LabelEncoder()
            label_encoder.fit(feature_vector[:, feature_index])
            label_encoders[feature_index] = label_encoder
        return label_encoders

    def __transform_single(self, x):
        x_copy = list(x)
        for feature_index in self.categorical_features:
            x_copy[feature_index] = self.encoders[feature_index].transform([x[feature_index]])[0]
        return x_copy

    def __inverse_transform_single(self, x):
        x_copy = list(x)
        for feature_index in self.categorical_features:
            x_copy[feature_index] = self.encoders[feature_index].inverse_transform([int(x[feature_index])])[0]
        return x_copy

    def transform(self, X):
        return np.array([self.__transform_single(x) for x in X])

    def inverse_transform(self, X):
        return [self.__inverse_transform_single(x) for x in X]

    def get_categorical_names(self):
        categorical_names = {}
        for index in self.categorical_features:
            categorical_names[index] = self.encoders[index].classes_
        return categorical_names


class PreProcess:
    # feature_vector should be an np.array with dtype=object if categorical features are present
    def __init__(self, feature_vector):
        self.categorical_features = self.get_categorical_features(feature_vector)
        self.has_categorical_features = len(self.categorical_features) != 0

        self.label_encoder = FeatureVectorLabelEncoder(feature_vector, self.categorical_features)
        train_label_encoded = self.label_encoder.transform(feature_vector)

        if self.has_categorical_features:
            self.one_hot_encoder = InvertibleOneHotEncoder(train_label_encoded, self.categorical_features)
            train_one_hot_encoded = self.one_hot_encoder.transform(train_label_encoded)
        else:
            train_one_hot_encoded = train_label_encoded

        self.scaler = StandardScaler()
        self.scaler.fit(train_one_hot_encoded)

        if self.has_categorical_features:
            for i in self.one_hot_encoder.active_features_:
                self.scaler.mean_[i] = 0
                self.scaler.scale_[i] = 1

    def transform(self, X, scale=False):
        result = self.label_encoder.transform(X)
        if self.has_categorical_features:
            result = self.one_hot_encoder.transform(result)
        if scale:
            result = self.scaler.transform(result)
        return result

    def inverse_transform(self, X, scale=False):
        result = X
        if scale:
            result = self.scaler.inverse_transform(result)
        if self.has_categorical_features:
            result = self.one_hot_encoder.inverse_transform(result)
        result = self.label_encoder.inverse_transform(result)
        return result

    @staticmethod
    def get_categorical_features(feature_vector):
        row = feature_vector[0]
        categorical_column_indices = []
        for i in range(len(row)):
            if type(row[i]) is str:
                categorical_column_indices.append(i)
        return categorical_column_indices


class Data:
    def __init__(self,
                 feature_vector,
                 target_vector,
                 feature_names,
                 target_vector_encoder=None,
                 pre_process_object=None):
        """
        Columns with values of type string will be interpreted as categorical features
        :param feature_vector: Should be of type np.array and with dtype=object
        :param target_vector
        :param feature_names
        :param target_vector_encoder: One relevant when using self.copy
        :param pre_process_object: One relevant when using self.copy
        """

        self.feature_vector = np.array(feature_vector)
        self.feature_names = np.array(feature_names)

        if target_vector_encoder is None:
            self.target_vector_encoder = LabelEncoder()
            self.target_vector = self.target_vector_encoder.fit_transform(target_vector)
        else:
            self.target_vector_encoder = target_vector_encoder
            self.target_vector = target_vector
        self.class_names = self.target_vector_encoder.classes_

        if pre_process_object is None:
            pre_process_object = PreProcess(feature_vector)
        self.pre_process_object = pre_process_object
        self.one_hot_encoded_feature_vector = self.pre_process(feature_vector, scale=False)
        self.scaled_feature_vector = self.pre_process_object.scaler.transform(self.one_hot_encoded_feature_vector)

    def pre_process(self, X, scale=True):
        return self.pre_process_object.transform(X, scale=scale)

    def inverse_pre_process(self, X, scale=True):
        return self.pre_process_object.inverse_transform(X, scale=scale)

    def copy(self, new_feature_vector, new_target_vector):
        return Data(new_feature_vector,
                    new_target_vector,
                    self.feature_names,
                    target_vector_encoder=self.target_vector_encoder,
                    pre_process_object=self.pre_process_object)

    def __len__(self):
        return len(self.feature_vector)


if __name__ == "__main__":
    feature_vector = np.array([[0, "a", "b", 0], [4, "u", "i", 1]], dtype=object)

    pre_process = PreProcess(feature_vector)

    new_instances = [[7.8, "a", "i", 9], [8, "u", "i", 7]]
    transformed = pre_process.transform(new_instances)
    inverse_transformed = pre_process.inverse_transform(transformed)
    t = 9
