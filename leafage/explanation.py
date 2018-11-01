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
        self.original_order_test_instance = pd.Series(test_instance, index=feature_names)

        self.local_model = local_model
        self.__sort_columns_according_to_importance()
        self.notebook_initialized = False
        self.plotly_imports_set = False

    def visualize_feature_importance(self, amount_of_features=5, target="write_to_file", path=None, show_values=False):
        """
        Visualize a bar plot which contains the most important features for the classification of self.test_instance
        :param amount_of_features: Amount of top features to include in the bar plot
        :param target: Should be either "notebook" or "write_to_file"
        :param path: If target="write_to_file", this parameter denotes where to save the image e.g. "../output/test.png"
        :param show_values: Shows the values of the instance being explained along each bar
        """
        self.__set_plotly_imports()
        figure = self.__visualize_feature_importance(amount_of_features, show_values=show_values)
        self.__export(figure, target, path)

    def visualize_examples(self, amount_of_features=5, target="write_to_file", path=None,
                           type="examples_in_support"):
        """
        Visualize the closest examples from the training-set
        :param amount_of_features: Amount of top features to include per example
        :param target: Denotes how to export the image. Should be either "notebook" or "write_to_file"
        :param path: If target="write_to_file", this parameter denotes where to save the image e.g. "../output/test.png"
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

    def visualize_leafage(self, amount_of_features=5, target="write_to_file", path=None, show_values=False):
        """
        Visualize closest examples and the feature importance from the training-set
        :param amount_of_features: Amount of top features to include per example
        :param target: Denotes how to export the image. Should be either "notebook" or "write_to_file"
        :param path: If target="write_to_file", this parameter denotes where to save the image e.g. "../output/test.png"
        :param show_values: Shows the values of the instance being explained along each bar
        """
        self.__set_plotly_imports()
        figure = self.__visualize_all(amount_of_features, show_values=show_values)
        self.__export(figure, target, path)

    def visualize_instance(self, target="write_to_file", path=None):
        """
        Visualize the instance being explained
        :param target: Denotes how to export the image. Should be either "notebook" or "write_to_file"
        :param path: If target="write_to_file", this parameter denotes where to save the image e.g. "../output/test.png"
        """
        self.__set_plotly_imports()
        figure = self.visualize_instance_one_line(self.original_order_test_instance)
        self.__export(figure, target, path)

    def visualize_prediction(self, target="write_to_file", path=None):
        """
        Visualize the prediction of the instance being explained
        :param target: Denotes how to export the image. Should be either "notebook" or "write_to_file"
        :param path: If target="write_to_file", this parameter denotes where to save the image e.g. "../output/test.png"
        """
        self.__set_plotly_imports()
        figure = self.__visualize_prediction()
        self.__export(figure, target, path)

    def __visualize_prediction(self):
        main_title = "Prediction: %s" % self.fact_class
        layout = go.Layout(title=main_title,
                           titlefont={"size": 32})
        figure = go.Figure(layout=layout)
        return figure

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

    def __sort_columns_according_to_importance(self, sort=False):
        if sort is True:
            sort_index = sorted(range(len(self.coefficients)), key=lambda i: abs(self.coefficients[i]), reverse=True)
        else:
            sort_index = range(len(self.feature_names))
        self.feature_names = self.feature_names[sort_index]
        self.coefficients = self.coefficients[sort_index]

        self.test_instance = pd.Series(self.test_instance[sort_index], index=self.feature_names)
        self.examples_in_support = pd.DataFrame(self.examples_in_support[:, sort_index], columns=self.feature_names)
        self.examples_against = pd.DataFrame(self.examples_against[:, sort_index], columns=self.feature_names)

    def __visualize_feature_importance(self, amount_of_features, show_values=False, as_sub_figure=False):

        color_examples_in_support = self.color_examples_in_support
        color_examples_against = self.color_examples_against
        color = "rgb(189,88,44,1)"

        if self.fact_class == "High":
            color_examples_in_support = self.color_examples_against
            color_examples_against = self.color_examples_in_support

        feature_names = self.feature_names[:amount_of_features]
        feature_values = self.test_instance.values[:amount_of_features]
        coefficients = self.coefficients[:amount_of_features]

        indices_positive = np.where(coefficients >= 0)
        indices_negative = np.where(coefficients < 0)
        coefficients = np.abs(coefficients)

        get_value = lambda feature, value: "<b>%s</b><br>%s" % (feature, value) if show_values else "<b>%s</b>" % feature

        x_values = np.array([get_value(i,j) for i, j in zip(feature_names, feature_values)])

        trace_positive = go.Bar(x=x_values[indices_positive],
                                y=coefficients[indices_positive],
                                #
                                marker=dict(color=color),
                                name="Supports %s" % self.fact_class)
        trace_negative = go.Bar(x=x_values[indices_negative],
                                y=coefficients[indices_negative],
                                #
                                marker=dict(color=color),
                                name="Supports %s" % self.foil_class)

        main_title = "Prediction: %s" % self.fact_class
        #sub_title = "The top %s most important features for the classification" % amount_of_features
        sub_title = "The importance of each feature for the prediction"

        if as_sub_figure:
            data = [trace_positive, trace_negative]
            layout = go.Layout(title=sub_title,
                              yaxis=dict(title='Importance'),
                              xaxis=dict(title="Features", categoryorder="array", categoryarray=x_values),
                              showlegend=False)
            fig = go.Figure(data=data, layout=layout)

        else:
            fig = tools.make_subplots(rows=1,
                                      cols=1,
                                      subplot_titles=[sub_title])

            fig.append_trace(trace_positive, 1, 1)
            fig.append_trace(trace_negative, 1, 1)

            fig["layout"].update(
                               yaxis=dict(title='Importance'),
                               xaxis=dict(title="Features", categoryorder="array", categoryarray=x_values),
                               showlegend=False,
                                # legend=dict(x=0.8,
                                #            y=1.0,
                                #            bgcolor='rgba(255, 255, 255, 0)',
                                #            bordercolor='rgba(255, 255, 255, 0)')
            )
            fig["layout"].update(title=main_title, titlefont={"size": 32})
            fig['layout'].update(margin=dict(t=100, l=50, r=50))

        return fig

    def __set_plotly_imports(self):
        if not self.plotly_imports_set:
            self.set_plotly_imports()
            self.plotly_imports_set = True

    @staticmethod
    def set_plotly_imports():
        global py, go, ff, iplot, download_plotlyjs, init_notebook_mode, tools, pio
        from plotly.offline import download_plotlyjs, iplot, init_notebook_mode
        import plotly.plotly as py
        import plotly.graph_objs as go
        import plotly.figure_factory as ff
        from plotly import tools

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
    def __visualize_table_ff(amount_of_features, df, header_background_color=None,
                             uneven_cell_color=None, even_cell_color=None):
        if header_background_color is None:
            header_background_color = "#00083e"
        if uneven_cell_color is None:
            uneven_cell_color = "#f2e5ff"
        if even_cell_color is None:
            even_cell_color = "#ffffff"

        df = df.iloc[:, 0:amount_of_features]

        df = df.astype(str)
        columns = list(df)
        max_amount = 135/len(columns)
        shorten = lambda row: [i[:max_amount - 4] + "..." if len(i) > max_amount else i for i in row]

        df.apply(shorten)
        df.columns = shorten(columns)

        color_scale = [[0, header_background_color], [.5, uneven_cell_color], [1, even_cell_color]]
        figure = ff.create_table(df, colorscale=color_scale)#, height_constant=60)

        figure.layout.width = 800

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
        fig["layout"].update(title="Prediction: %s" % self.fact_class, titlefont={"size": 32})

        fig.append_trace(table_in_support['data'][0], 1, 1)
        fig.append_trace(table_against['data'][0], 2, 1)

        fig['layout']['xaxis1'] = dict(fig['layout']['xaxis1'], **table_in_support['layout']['xaxis'])
        fig['layout']['yaxis1'] = dict(fig['layout']['yaxis1'], **table_in_support['layout']['yaxis'])
        fig['layout']['xaxis2'] = dict(fig['layout']['xaxis2'], **table_against['layout']['xaxis'])
        fig['layout']['yaxis2'] = dict(fig['layout']['yaxis2'], **table_against['layout']['yaxis'])

        for k in range(len(table_against['layout']['annotations'])):
            table_against['layout']['annotations'][k].update(xref='x2', yref='y2')

        fig['layout']['annotations'].extend(table_in_support['layout']['annotations'] + table_against['layout']['annotations'])

        fig['layout'].update(width=820, height=600, margin=dict(t=100, l=50, r=50, b=20))

        return fig

    def __visualize_all(self, amount_of_features, show_values=False):
        table_in_support = self.__visualize_table_ff(amount_of_features, self.examples_in_support,
                                                            self.color_examples_in_support)
        table_against = self.__visualize_table_ff(amount_of_features, self.examples_against,
                                                     self.color_examples_against)

        feature_importance = self.__visualize_feature_importance(amount_of_features=amount_of_features,
                                                                 show_values=show_values,
                                                                 as_sub_figure=True)

        title_in_support = 'Most similar houses with value %s' % self.fact_class
        title_against = 'Most similar houses with value %s' % self.foil_class
        title_feature_importance = "The %s most important features for the prediction" % amount_of_features

        fig = tools.make_subplots(specs=[[{'rowspan':2, 'colspan': 2}, None, {'colspan': 3}, None, None],
                                         [None, None, {'colspan': 3}, None, None]],
                                  rows=2,
                                  cols=5,
                                  subplot_titles=(title_feature_importance, title_in_support, title_against),
                                  vertical_spacing=0.085)
        fig["layout"].update(title="Prediction: %s" % self.fact_class, titlefont={"size": 32})
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
        fig.append_trace(feature_importance["data"][1], 1, 1)

        fig['layout']['xaxis1'] = dict(fig['layout']['xaxis1'], **feature_importance['layout']['xaxis'])
        fig['layout']['yaxis1'] = dict(fig['layout']['yaxis1'], **feature_importance['layout']['yaxis'])
        fig["layout"]["showlegend"] = False

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

        fig['layout'].update(width=1300, height=600, margin=dict(t=100, l=50, r=50, b=100))

        return fig

    @staticmethod
    def visualize_instance_one_line(pd_series):
        df = pd.DataFrame([pd_series.values], columns=pd_series.index)
        header = "#c2bc80"
        uneven_cell = "#e9e7d8"
        figure = Explanation.__visualize_table_ff(len(pd_series), df, header_background_color=header, uneven_cell_color=uneven_cell)
        return figure

    @staticmethod
    def __visualize_instance_multiple_lines(pd_series):
        header_background_color = "rgb(47, 80, 135, 1)"
        cell_background_color = "rgb(182, 187, 196, 1)"

        num_columns = 7
        d = len(pd_series)
        num_rows = int(np.ceil(d/float(num_columns)))

        extra_cells = (num_columns - (d % num_columns)) % num_columns
        total_cells = num_columns*num_rows*2

        header = Explanation.__make_bold(pd_series.index)
        values = pd_series.values

        header_values = np.append(header, [""]*extra_cells).reshape(num_rows, num_columns)
        cell_values = np.append(values, [""]*extra_cells).reshape(num_rows, num_columns)
        all_cells = np.array(zip(header_values, cell_values)).reshape((-1, num_columns)).transpose()

        values_per_cel = lambda h,c: np.array(zip([[h]*num_columns]*total_cells,
                                                  [[c]*num_columns]*total_cells))\
                                              .reshape((-1, num_columns))\
                                              .transpose()
        fill_color = values_per_cel(header_background_color, cell_background_color)
        font_color = values_per_cel("white", "black")
        font_size = values_per_cel(14, 14)

        trace = go.Table(
                        header=dict(line=dict(color='white')),
                        cells=dict(values=all_cells,
                                   align="center",
                                   line=dict(color="rgb(207, 210, 214)"),
                                   fill=dict(color=fill_color),
                                   font=dict(color=font_color, size=font_size),
                                   height=30
                                   )
                        )
        layout = go.Layout(title="A house in the market",
                           width=1300,
                           height=800,
                           margin=dict(t=100, l=50, r=50, b=100))
        figure = go.Figure(data=[trace], layout=layout)
        return figure

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
