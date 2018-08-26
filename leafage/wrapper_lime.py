from lime import lime_tabular
import numpy as np

from local_model import LinearModel


class WrapperLime:
    random_state = 11

    def __init__(self, training_data, predict_proba):
        self.training_data = training_data

        categorical_features = training_data.pre_process_object.categorical_features
        self.label_encoder = training_data.pre_process_object.label_encoder

        self.classes = sorted(np.unique(training_data.target_vector))

        # Label encode the feature vector, lime only accepts numbers
        data_set = self.label_encoder.transform(training_data.feature_vector)
        categorical_names = self.label_encoder.get_categorical_names()

        # Before feeding to the classifier the categorical values have to be one-hot-encoded
        pre_process = training_data.pre_process_object
        if pre_process.has_categorical_features:
            self.predict_proba = lambda x: predict_proba(pre_process.one_hot_encoder.transform(x))
        else:
            self.predict_proba = predict_proba

        np.random.seed(self.random_state)
        self.explainer = lime_tabular.LimeTabularExplainer(data_set,
                                                           feature_names=self.training_data.feature_names,
                                                           discretize_continuous=False,
                                                           class_names=self.training_data.class_names,
                                                           categorical_features=categorical_features,
                                                           categorical_names=categorical_names)

    def get_local_model(self, instance, _):
        instance = self.label_encoder.transform([instance])[0]
        local_model = self.explainer.explain_instance(instance,
                                                      self.predict_proba,
                                                      num_features=len(instance))
        coef = map(lambda t: t[1], sorted(local_model.local_exp[1], key=lambda x: x[0]))
        intercept = local_model.intercept
        pre_process = lambda X: self.explainer.scaler.transform(self.label_encoder.transform(X))
        return LinearModel(coef, intercept[1], pre_process, threshold=0.5, classes=self.classes)
