from sklearn.preprocessing import StandardScaler


class PreProcess:
    def __init__(self, feature_vector, preprocessing_method="z_normalization"):
        self.method = preprocessing_method

        if preprocessing_method == "z_normalization":
            self.pre_process_basis = StandardScaler()
            self.pre_process_basis.fit(feature_vector)
            self.transform = self.pre_process_basis.transform
            self.inverse_transform = self.pre_process_basis.inverse_transform
        else:
            self.transform = lambda x: x
            self.inverse_transform = lambda x: x


class Data:
    def __init__(self,
                 feature_vector,
                 target_vector,
                 feature_names,
                 class_names,
                 categorical_features=[],
                 categorical_names={},
                 encoder=None,
                 preprocessing_method="z_normalization"):
        self.feature_vector = feature_vector
        self.target_vector = target_vector
        self.feature_names = feature_names
        self.class_names = class_names
        self.categorical_features = categorical_features
        self.categorical_names = categorical_names
        self.encoder = encoder
        self.preprocessing_method = preprocessing_method

        self.pre_process_object = None
        self.pre_processed_feature_vector = None
        self.set_pre_process_method(preprocessing_method)

    def set_pre_process_method(self, method):
        self.pre_process_object = PreProcess(self.feature_vector, method)
        self.pre_processed_feature_vector = None

    def pre_process(self, X):
        return self.pre_process_object.transform(X)

    def inverse_pre_process(self, X):
        return self.pre_process_object.inverse_transform(X)

    def get_pre_processed_feature_vector(self):
        if self.pre_processed_feature_vector is None:
            self.pre_processed_feature_vector = self.pre_process(self.feature_vector)
        return self.pre_processed_feature_vector

    def copy(self, new_feature_vector, new_target_vector):
        return Data(new_feature_vector,
                    new_target_vector,
                    self.feature_names,
                    self.class_names,
                    self.categorical_features,
                    self.categorical_names,
                    self.encoder,
                    self.preprocessing_method)

    def __len__(self):
        return len(self.feature_vector)