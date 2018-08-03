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

    def visualize_feature_importance(self, amount_of_features=5, target="notebook", path=None):
        """
        Visualize a bar plot which contains the most important features for the classification of self.test_instance
        :param amount_of_features: Amount of top features to include in the bar plot
        :param target: Should be either "notebook" or "write_to_file"
        :param path: If target="to_file", this parameter denotes where to save the image e.g. "../output/test.png"
        """
        figure = self.__visualize_feature_importance(amount_of_features)
        self.__export(figure, target, path)

    def visualize_examples(self, amount_of_features=5, target="notebook", path=None, type="examples_in_support"):
        """
        Visualize a table of closest examples from the training-set
        :param amount_of_features:
        :param target: Denotes how to export the image. Should be either "notebook" or "write_to_file"
        :param path: If target="to_file", this parameter denotes where to save the image e.g. "../output/test.png"
        :param type: Should be either "examples_in_support" or "examples_against"
        """
        if type == "examples_in_support":
            hc = 'rgba(0,184,0,1)'
            cc = 'rgba(143,255,143,1)'
            figure = self.__visualize_table(amount_of_features, self.examples_in_support, self.fact_class,
                                            header_background_color=hc, cell_background_color=cc)
        elif type == "examples_against":
            hc = 'rgba(255,20,20,1)'
            cc = 'rgba(255,170,170,1)'
            figure = self.__visualize_table(amount_of_features, self.examples_against, self.foil_class,
                                            header_background_color=hc, cell_background_color=cc)
        else:
            raise ValueError("%s not supported" % target)

        self.__export(figure, target, path)

    def __export(self, figure, target, path):
        if target == "notebook":
            self.__visualize_notebook(figure)
        elif target == "write_to_file":
            if path is None:
                raise ValueError("Argument path has to be non null")
            else:
                self.__visualize_png(figure, path)
        else:
            raise ValueError("%s not supported" % target)

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

    @staticmethod
    def __visualize_table(amount_of_features, df, class_name, header_background_color=None, cell_background_color=None):
        if header_background_color is None:
            header_background_color = 'rgba(0,184,0,1)'
        if cell_background_color is None:
            cell_background_color = 'rgba(143,255,143,1)'

        header_values = Explanation.__make_bold(list(df)[:amount_of_features])
        cell_values = df.values.transpose()[:amount_of_features]

        trace = go.Table(
            header=dict(values=header_values,
                        fill=dict(color=header_background_color),
                        align="center",
                        font=dict(color='white', size=15),
                        line=dict(color="white", width=3)),
            cells=dict(values=cell_values,
                       fill=dict(color=cell_background_color),
                       align="center",
                       font=dict(size=12),
                       line=dict(color="white")))
        layout = go.Layout(title="<b>Examples in support of value %s</b>" % class_name)

        figure = go.Figure(data=[trace], layout=layout)
        return figure

    @staticmethod
    def __make_bold(container):
        return ["<b>%s</b>" % x for x in container]

    @staticmethod
    def __visualize_notebook(figure):
        py.iplot(figure)

    @staticmethod
    def __visualize_png(figure, path):
        py.image.save_as(figure, filename=path)
        print("Image saved as %s" % path)


