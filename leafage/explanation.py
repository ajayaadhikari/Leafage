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

    def __visualize_feature_importance(self, amount_of_features):
        feature_names = self.feature_names[:amount_of_features]
        feature_values = self.test_instance.values[:amount_of_features]
        coefficients = self.coefficients[:amount_of_features]

        indices_positive = np.where(coefficients >= 0)
        indices_negative = np.where(coefficients < 0)
        coefficients = np.abs(coefficients)

        get_x_values = lambda indices: ["<b>%s</b><br><b>%s</b>" % (i,j) for i,j in zip(feature_names[indices], feature_values[indices])]

        trace_positive = go.Bar(x=get_x_values(indices_positive),
                                y=coefficients[indices_positive],
                                marker=dict(color="green"),
                                name="Supports %s" % self.fact_class)
        trace_negative = go.Bar(x=get_x_values(indices_negative),
                                y=coefficients[indices_negative],
                                marker=dict(color="red"),
                                name="Supports %s" % self.foil_class)

        data = [trace_positive, trace_negative]
        layout = go.Layout(title="The top %s most important features for the classification" % amount_of_features,
                           yaxis=dict(title='Importance'),
                           xaxis=dict(title="Features", categoryorder="array", categoryarray=feature_names))
        figure = go.Figure(data=data, layout=layout)
        return figure

    def visualize_feature_importance_notebook(self, amount_of_features=5):
        figure = self.__visualize_feature_importance(amount_of_features)
        py.iplot(figure)

    def visualize_feature_importance_png(self, path, amount_of_features=5):
        figure = self.__visualize_feature_importance(amount_of_features)
        py.image.save_as(figure, filename=path)
