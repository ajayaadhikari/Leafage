import matplotlib.pyplot as plt
import numpy as np
import random

from sklearn.exceptions import UndefinedMetricWarning

from faithfulness import Faithfulness
from leafage import LeafageBinaryClass
from leafage.use_cases.data import Data

random_state = 9


class Line:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def __call__(self, x):
        """
        input: f(x,y) = ax + by + c
        output: y = -(a/b)*x + -c/b
        """
        return -((self.a/float(self.b))*x + self.c/float(self.b))


class Parabola:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def __call__(self, x):
        """
        :return: f(x) = a*x^2 + b*x + c
        """
        return self.a * (x ** 2) + self.b * x + self.c


class PlotLocalModel:
    def __init__(self, x_spacing, y_spacing, local_model, contour_type="standard"):
        self.local_model = local_model
        self.contour_type = contour_type
        self.x_spacing = x_spacing
        self.y_spacing = y_spacing

        self.X, self.Y = np.meshgrid(x_spacing, y_spacing)

    def get_unbiased_distance(self):
        return self.perform_pair_wise(self.X,
                                      self.Y,
                                      lambda x, y: self.local_model.distances.get_unbiased_distance([x, y]))

    def get_weights(self):
        return self.perform_pair_wise(self.X,
                                      self.Y,
                                      lambda x, y: self.local_model.neighbourhood.get_weight([x, y]))

    def get_black_box_distance(self):
        return self.perform_pair_wise(self.X,
                                      self.Y,
                                      lambda x, y: self.local_model.distances.get_black_box_distance([x, y]))

    def get_final_distance(self):
        return self.perform_pair_wise(self.get_unbiased_distance(),
                                      self.get_black_box_distance(),
                                      lambda x, y: x + y)

    @staticmethod
    def perform_pair_wise(matrix1, matrix2, function_):
        result = []
        for i in range(len(matrix1)):
            result.append([])
            for j in range(len(matrix1[0])):
                result[i].append(function_(matrix1[i][j], matrix2[i][j]))
        return np.array(result)

    def plot_distance_function(self, distance_type="unbiased_distance"):
        if distance_type == "unbiased_distance":
            Z = self.get_unbiased_distance()
        elif distance_type == "weights":
            Z = self.get_weights()
        elif distance_type == "black_box_distance":
            Z = self.get_black_box_distance()
        else:
            Z = self.get_final_distance()

        self.plot_contour(Z)

    def plot_distance_function_2(self, distance_function):
        Z = self.perform_pair_wise(self.X,
                                   self.Y,
                                   distance_function)
        self.plot_contour(Z)

    def plot_contour(self, Z):
        if self.contour_type == "standard":
            plt.contour(self.X, self.Y, Z, colors='black')
        elif self.contour_type == "color":
            plt.contourf(self.X, self.Y, Z, 20, cmap='RdGy')
            plt.colorbar()

    def plot_local_model(self, type="original"):
        if type == "moved":
            intercept = self.local_model.linear_model.moved_intercept
        else:
            intercept = self.local_model.linear_model.original_intercept

        line = Line(self.local_model.linear_model.coefficients[0],
                    self.local_model.linear_model.coefficients[1],
                    intercept)
        self.draw_curve(line)

    def draw_curve(self, curve):
        y = [curve(t) for t in self.x_spacing]
        plt.plot(self.x_spacing, y)


class TwoDimensionExample:
    # Plot ranges
    xmin = -15
    xmax = 15
    ymin = -5
    ymax = 25

    # Parabola parameter
    a = -1
    b = 0
    c = 25
    black_box_curve = Parabola(a, b, c)

    slack = 2

    def __init__(self):
        plt.figure()
        plt.xlim([self.xmin - self.slack, self.xmax + self.slack])
        plt.ylim([self.ymin - self.slack, self.ymax + self.slack])
        random.seed(random_state)

        self.amount_of_points = (self.xmax-self.xmin)*2 + (self.ymax-self.ymin)*2
        self.amount_per_unit = 2
        self.x_spacing = np.linspace(self.xmin, self.xmax, (self.xmax - self.xmin) * self.amount_per_unit)
        self.y_spacing = np.linspace(self.ymin, self.ymax, (self.ymax - self.ymin) * self.amount_per_unit)

        self.points = self.sample_points()
        self.labels = self.get_labels()

    def get_data(self):
        return Data(self.points, self.labels, ["x", "y"], ["red", "black"], preprocessing_method=None)

    def sample_points(self):
        x_range = self.xmax - self.xmin
        y_range = self.ymax - self.ymin
        result = []
        for _ in range(self.amount_of_points):
            result.append([self.xmin + x_range * random.random(), self.ymin + y_range * random.random()])
        return np.array(result)

    def get_labels(self):
        return np.array([self.get_label(x) for x in self.points])

    def get_label(self, point):
        if point[1] > self.black_box_curve(point[0]):
            return 1
        else:
            return 0

    def plot_setting(self):
        self.plot_black_box_curve()
        self.plot_training_points()

    def plot_curve(self, curve):
        y = [curve(t) for t in self.x_spacing]
        plt.plot(self.x_spacing, y)

    @staticmethod
    def plot_points(points, labels):
        ones_index = filter(lambda index: labels[index] == 1, range(len(labels)))
        zeros_index = filter(lambda index: labels[index] == 0, range(len(labels)))

        get_index = lambda index, container: map(lambda element: element[index], container)
        x = np.array(get_index(0, points))
        y = np.array(get_index(1, points))

        plt.plot(x[zeros_index], y[zeros_index], "ro", x[ones_index], y[ones_index], "g^")

    @staticmethod
    def plot_point(point):
        plt.plot(point[0], point[1], "y^")

    def plot_training_points(self):
        self.plot_points(self.points, self.labels)

    def plot_black_box_curve(self):
        self.plot_curve(self.black_box_curve)

    def plot_local_model(self, test_point):
        #self.plot_points()
        self.plot_black_box_curve()

        leafage = LeafageBinaryClass(self.get_data(), self.labels, random_state)
        local_model = leafage.explain(test_point, self.get_label(test_point)).local_model

        plot_local_model = PlotLocalModel(self.x_spacing, self.y_spacing, local_model)
        self.plot_point(test_point)
        self.plot_points(local_model.neighbourhood.instances, local_model.neighbourhood.labels)
        #plot_local_model.plot_distance_function("weights")
        plot_local_model.plot_local_model()

    def test_evaluation(self):
        #train, test, labels_train, labels_test = train_test_split(self.points, self.labels, train_size=0.5)
        leafage = LeafageBinaryClass(self.get_data(), self.labels, random_state)
        evaluation = Faithfulness(self.points, self.labels, leafage.get_local_model, np.arange(0.36, 1, 0.05))

        evaluation.evaluate(self)

    def vary_distance_to_boundary(self):
        #Get the local linear boundary

        leafage = LeafageBinaryClass(self.get_data(), self.labels, random_state)
        explanation = []
        range_ = np.arange(self.xmin, self.xmax,0.5)
        for i in range_:
            test_point = [i, 3.7]
            explanation.append(leafage.explain(test_point, self.get_label(test_point)).local_model)
        #
        distances = []
        sigma = []
        for e in explanation:
           distances.append(e.neighbourhood.get_distance_to_closest_opposite_instance(e.distances.get_unbiased_distance))
           sigma.append(e.sigma)
        #
        plt.scatter(range_, sigma)
        plt.title("Moving point vertically from -15 to -3")
        plt.xlabel('Euclidean distance to the closet boundary')
        plt.ylabel('Optimal sigma')

    def plot_contours(self):
        points = [[-12, -3], [-7, 7], [-1, 15], [10, 5]]

        leafage = LeafageBinaryClass(self.get_data(), self.labels, random_state)
        #self.plot_training_points()
        self.plot_black_box_curve()

        for point in points:
            local_model = leafage.explain(point, self.get_label(point), t=self).local_model
            contour = PlotLocalModel(self.x_spacing, self.y_spacing, local_model)
            self.plot_local_model(point)
            contour.plot_local_model("original")

            # Plot the test point
            #c1 = plt.Circle(point, local_model.get_diameter(), fill=False)
            #fig = plt.gcf()
            #ax = fig.gca()
##
            #ax.add_artist(c1)
            #contour.plot_local_model()


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    a = TwoDimensionExample()
    a.test_evaluation()

    plt.show()
