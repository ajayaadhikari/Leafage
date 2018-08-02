import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np

class Explanation:
    def __init__(self,
                 test_instance,
                 examples_in_support,
                 examples_against,
                 coefficients,
                 fact_class,
                 foil_class,
                 feature_names,
                 local_model):

        self.test_instance = test_instance
        self.examples_in_support = examples_in_support
        self.examples_against = examples_against
        self.coefficients = coefficients
        self.fact_class = fact_class
        self.foil_class = foil_class
        self.feature_names = feature_names

        self.local_model = local_model
        self.__sort_columns_according_to_importance()

    def __sort_columns_according_to_importance(self):
        sort_index = sorted(range(len(self.coefficients)), key=lambda i: abs(self.coefficients[i]), reverse=True)
        self.feature_names = self.feature_names[sort_index]
        self.coefficients = self.coefficients[sort_index]

        self.test_instance = pd.Series(self.test_instance[sort_index], index=self.feature_names)
        self.examples_in_support = pd.DataFrame(self.examples_in_support[:, sort_index], columns=self.feature_names)
        self.examples_against = pd.DataFrame(self.examples_against[:, sort_index], columns=self.feature_names)

    #def visualize_bar_plot(self, amount_of_features):
    #    data = [go.Bar(
#
    #    )]
#