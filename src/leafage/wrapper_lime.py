import sklearn.ensemble
from lime import lime_tabular
import numpy as np

from src.leafage.local_model import LinearModel
from src.use_cases.adult import Adult


class WrapperLime:
    random_state = 11

    def __init__(self, training_data, predict_proba, classes):
        self.training_data = training_data
        self.predict_proba = predict_proba
        self.classes = classes

        np.random.seed(self.random_state)
        self.explainer = lime_tabular.LimeTabularExplainer(self.training_data.feature_vector,
                                                           feature_names=self.training_data.feature_names,
                                                           discretize_continuous=False,
                                                           class_names=self.training_data.class_names,
                                                           categorical_names=self.training_data.categorical_names,
                                                           categorical_features=self.training_data.categorical_features)

    def get_local_model(self, instance):
        local_model = self.explainer.explain_instance(instance,
                                                      self.predict_proba,
                                                      num_features=len(instance))
        coef = map(lambda t: t[1], sorted(local_model.local_exp[1], key=lambda x: x[0]))
        intercept = local_model.intercept
        return LinearModel(coef, intercept[1], self.explainer.scaler, threshold=0.5, classes=self.classes)
