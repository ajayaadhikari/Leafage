import pandas as pd
import numpy as np


class Explanation:
    color_examples_in_support = 'rgba(0,184,0,1)'
    color_examples_against = 'rgba(255,20,20,1)'

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
        self.notebook_initialized = False
        self.plotly_imports_set = False

    def visualize_feature_importance(self, amount_of_features=5, target="notebook", path=None):
        """
        Visualize a bar plot which contains the most important features for the classification of self.test_instance
        :param amount_of_features: Amount of top features to include in the bar plot
        :param target: Should be either "notebook" or "write_to_file"
        :param path: If target="to_file", this parameter denotes where to save the image e.g. "../output/test.png"
        """
        self.__set_plotly_imports()
        figure = self.__visualize_feature_importance(amount_of_features)
        self.__export(figure, target, path)

    def visualize_examples(self, amount_of_features=5, target="notebook", path=None, type="examples_in_support"):
        """
        Visualize a table of closest examples from the training-set
        :param amount_of_features: Amount of top features to include per example
        :param target: Denotes how to export the image. Should be either "notebook" or "write_to_file"
        :param path: If target="to_file", this parameter denotes where to save the image e.g. "../output/test.png"
        :param type: Should be either "examples_in_support" or "examples_against" or "both"
        """
        self.__set_plotly_imports()
        if type == "examples_in_support":
            figure = self.__visualize_table_ff(amount_of_features, self.examples_in_support, self.color_examples_in_support)
        elif type == "examples_against":
            figure = self.__visualize_table_ff(amount_of_features, self.examples_against, self.color_examples_against)
        elif type == "both":
            figure = self.__visualize_examples(amount_of_features)
        else:
            raise ValueError("%s not supported" % target)

        self.__export(figure, target, path)

    def visualize_leafage(self, amount_of_features=5, path=None):
        self.__set_plotly_imports()
        figure = self.__visualize_all(amount_of_features)
        self.__export(figure, "write_to_file", path)

    def __export(self, figure, target, path):
        if target == "notebook":
            self.__visualize_notebook(figure)
        elif target == "write_to_file":
            if path is None:
                raise ValueError("Argument path has to be non null")
            else:
                self.visualize_png(figure, path)
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

        x_values = np.array(["<b>%s</b><br>%s" % (i,j) for i,j in zip(feature_names, feature_values)])

        trace_positive = go.Bar(x=x_values[indices_positive],
                                y=coefficients[indices_positive],
                                marker=dict(color="green"),
                                name="Supports %s" % self.fact_class)
        trace_negative = go.Bar(x=x_values[indices_negative],
                                y=coefficients[indices_negative],
                                marker=dict(color="red"),
                                name="Supports %s" % self.foil_class)

        data = [trace_positive, trace_negative]
        layout = go.Layout(title="The top %s most important features for the classification" % amount_of_features,
                           yaxis=dict(title='Importance'),
                           xaxis=dict(title="Features", categoryorder="array", categoryarray=x_values))
        figure = go.Figure(data=data, layout=layout)
        return figure

    def __set_plotly_imports(self):
        if not self.plotly_imports_set:
            global py, go, ff, iplot, download_plotlyjs, init_notebook_mode, tools
            from plotly.offline import download_plotlyjs, iplot, init_notebook_mode
            import plotly.plotly as py
            import plotly.graph_objs as go
            import plotly.figure_factory as ff
            from plotly import tools
            self.plotly_imports_set = True

    def to_json(self):
        jsonized = {'coefficients': list(self.coefficients),
                    'examples_against': self.examples_against.to_json(),
                    'examples_in_support': self.examples_in_support.to_json(),
                    'feature_names': list(self.feature_names),
                    'fact_class': self.fact_class,
                    'foil_class': self.foil_class
                    }
        return jsonized

    @staticmethod
    def __visualize_table_ff(amount_of_features, df, header_background_color):
        df = df.iloc[:, 0:amount_of_features]

        df = df.astype(str)
        columns = list(df)
        max_amount = 135/len(columns)
        shorten = lambda row: [i[:max_amount - 4] + "..." if len(i) > max_amount else i for i in row]

        df.apply(shorten)
        df.columns = shorten(columns)

        color_scale = [[0, header_background_color], [.5, '#f2e5ff'], [1, '#ffffff']]
        figure = ff.create_table(df, colorscale=color_scale, height_constant=60)

        figure.layout.width = 1000

        for i in range(len(figure.layout.annotations)):
            figure.layout.annotations[i].font.size = 13

        return figure

    def __visualize_examples(self, amount_of_features):
        table_in_support = self.__visualize_table_ff(amount_of_features, self.examples_in_support,
                                                            self.color_examples_in_support)
        table_against = self.__visualize_table_ff(amount_of_features, self.examples_against,
                                                         self.color_examples_against)

        title_in_support = 'Examples in support of prediction <b>%s</b>' % self.fact_class
        title_against = 'Most relevant counter-examples from class <b>%s</b>' % self.foil_class

        fig = tools.make_subplots(rows=2,
                                  cols=1,
                                  print_grid=True,
                                  vertical_spacing=0.085,
                                  subplot_titles=(title_in_support, title_against))

        fig.append_trace(table_in_support['data'][0], 1, 1)
        fig.append_trace(table_against['data'][0], 2, 1)

        fig['layout']['xaxis1'] = dict(fig['layout']['xaxis1'], **table_in_support['layout']['xaxis'])
        fig['layout']['yaxis1'] = dict(fig['layout']['yaxis1'], **table_in_support['layout']['yaxis'])
        fig['layout']['xaxis2'] = dict(fig['layout']['xaxis2'], **table_against['layout']['xaxis'])
        fig['layout']['yaxis2'] = dict(fig['layout']['yaxis2'], **table_against['layout']['yaxis'])

        for k in range(len(table_against['layout']['annotations'])):
            table_against['layout']['annotations'][k].update(xref='x2', yref='y2')

        fig['layout']['annotations'].extend(table_in_support['layout']['annotations'] + table_against['layout']['annotations'])

        fig['layout'].update(width=800, height=600, margin=dict(t=100, l=50, r=50, b=50))

        return fig

    def __visualize_all(self, amount_of_features):
        table_in_support = self.__visualize_table_ff(amount_of_features, self.examples_in_support,
                                                            self.color_examples_in_support)
        table_against = self.__visualize_table_ff(amount_of_features, self.examples_against,
                                                         self.color_examples_against)
        feature_importance = self.__visualize_feature_importance(amount_of_features)

        title_in_support = 'Examples in support of prediction <b>%s</b>' % self.fact_class
        title_against = 'Most relevant counter-examples from class <b>%s</b>' % self.foil_class

        fig = tools.make_subplots(specs=[[{'rowspan':2, 'colspan': 2}, None, {'colspan': 3}, None, None],
                                         [None, None, {'colspan': 3}, None, None]],
                                  rows=2,
                                  cols=5,
                                  subplot_titles=("lol", title_in_support, title_against),
                                  vertical_spacing=0.085)

        for i in range(len(table_in_support.data)):
            table_in_support.data[i].xaxis = 'x2'
            table_in_support.data[i].yaxis = 'y2'

        for i in range(len(table_against.data)):
            table_against.data[i].xaxis = 'x3'
            table_against.data[i].yaxis = 'y3'

        for i in range(len(feature_importance.data)):
            feature_importance.data[i].xaxis = 'x1'
            feature_importance.data[i].yaxis = 'y1'

        fig.append_trace(table_in_support['data'][0], 1, 3)
        fig.append_trace(table_against['data'][0], 2, 3)
        fig.append_trace(feature_importance["data"][0], 1, 1)

        fig['layout']['xaxis1'] = dict(fig['layout']['xaxis1'], **feature_importance['layout']['xaxis'])
        fig['layout']['yaxis1'] = dict(fig['layout']['yaxis1'], **feature_importance['layout']['yaxis'])

        fig['layout']['xaxis2'] = dict(fig['layout']['xaxis2'], **table_in_support['layout']['xaxis'])
        fig['layout']['yaxis2'] = dict(fig['layout']['yaxis2'], **table_in_support['layout']['yaxis'])
        fig['layout']['xaxis3'] = dict(fig['layout']['xaxis3'], **table_against['layout']['xaxis'])
        fig['layout']['yaxis3'] = dict(fig['layout']['yaxis3'], **table_against['layout']['yaxis'])

        for k in range(len(feature_importance['layout']['annotations'])):
            feature_importance['layout']['annotations'][k].update(xref='x1', yref='y1')

        for k in range(len(table_in_support['layout']['annotations'])):
            table_in_support['layout']['annotations'][k].update(xref='x2', yref='y2')

        for k in range(len(table_against['layout']['annotations'])):
            table_against['layout']['annotations'][k].update(xref='x3', yref='y3')

        fig['layout']['annotations'].extend(feature_importance['layout']['annotations'] +
                                            table_in_support['layout']['annotations'] +
                                            table_against['layout']['annotations'])

        fig['layout'].update(width=1000, height=600, margin=dict(t=100, l=50, r=50, b=50))

        return fig

    def __combine_tables(self, amount_of_features):

        trace_in_support = self.__visualize_table_trace(amount_of_features, self.examples_in_support, "in_support")
        trace_against = self.__visualize_table_trace(amount_of_features, self.examples_against, "against")

        fig = tools.make_subplots(rows=1, cols=2)

        fig.append_trace(trace_in_support, 1, 1)
        fig.append_trace(trace_against, 1, 2)

        fig['layout'].update(height=600, width=800, title='i <3 annotations and subplots')

        py.image.save_as(fig, filename="lol.png")

        return fig

    @staticmethod
    def __visualize_table_trace(amount_of_features, df, type="in_support"):

        if type == "in_support":
            header_background_color = "rgba(0, 184, 0, 1)"
            cell_background_color = 'rgba(143,255,143,1)'
        elif type == "against":
            header_background_color = 'rgba(255,20,20,1)'
            cell_background_color = 'rgba(255,170,170,1)'
        else:
            raise ValueError("Type %s not supported" % type)

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
        return trace

    @staticmethod
    def __make_bold(container):
        return ["<b>%s</b>" % x for x in container]

    def __visualize_notebook(self, figure):
        if not self.notebook_initialized:
            init_notebook_mode()
            self.notebook_initialized = True
        iplot(figure)

    @staticmethod
    def visualize_png(figure, path):
        py.image.save_as(figure, filename=path)
        print("Image saved as %s" % path)


def test_subplot():
    from plotly.offline import download_plotlyjs, iplot, init_notebook_mode
    import plotly.plotly as py
    import plotly.figure_factory as ff
    from plotly import tools

    data_US = [['Country', 'Year', 'Population'],
               ['United States', 2000, 282200000],
               ['United States', 2005, 295500000],
               ['United States', 2010, 309000000]
               ]
    data_Canada = [['Country', 'Year', 'Population'],
                   ['Canada', 2000, 27790000],
                   ['Canada', 2005, 32310000],
                   ['Canada', 2010, 34000000]]

    table1 = ff.create_table(data_US)
    table2 = ff.create_table(data_Canada)

    fig = tools.make_subplots(rows=2,
                              cols=1,
                              print_grid=True,
                              vertical_spacing=0.085,
                              subplot_titles=('US population', 'Canada population')
                              )

    fig.append_trace(table1['data'][0], 1, 1)
    fig.append_trace(table2['data'][0], 2, 1)

    fig['layout']['xaxis1'] = dict(fig['layout']['xaxis1'], **table1['layout']['xaxis'])
    fig['layout']['yaxis1'] = dict(fig['layout']['yaxis1'], **table1['layout']['yaxis'])
    fig['layout']['xaxis2'] = dict(fig['layout']['xaxis2'], **table2['layout']['xaxis'])
    fig['layout']['yaxis2'] = dict(fig['layout']['yaxis2'], **table2['layout']['yaxis'])

    for k in range(len(table2['layout']['annotations'])):
        table2['layout']['annotations'][k].update(xref='x2', yref='y2')

    fig['layout']['annotations'].extend(table1['layout']['annotations'] + table2['layout']['annotations'])

    fig['layout'].update(width=800, height=600, margin=dict(t=100, l=50, r=50, b=50))

    py.image.save_as(fig, filename="lol2.png")

if __name__ == "__main__":
    test_subplot()