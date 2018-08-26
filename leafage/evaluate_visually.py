import matplotlib.pyplot as plt
import numpy as np
import random

from sklearn.datasets import make_classification
from sklearn.exceptions import UndefinedMetricWarning

from faithfulness import Faithfulness
from leafage import LeafageBinary
from wrapper_lime import WrapperLime
from use_cases.data import Data, PreProcess

from utils.Classifiers import train

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
    def __init__(self, x_spacing, y_spacing, local_model, contour_type="color"):
        self.local_model = local_model
        self.contour_type = contour_type
        self.x_spacing = x_spacing
        self.y_spacing = y_spacing

        self.X, self.Y = np.meshgrid(x_spacing, y_spacing)

    def get_biased_distance(self):
        return self.perform_pair_wise(self.X,
                                      self.Y,
                                      lambda x, y: self.local_model.distances.get_biased_distance([x, y]))

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
        return self.perform_pair_wise(self.X,
                                      self.Y,
                                      lambda x, y: self.local_model.distances.get_final_distance([x, y]))

    @staticmethod
    def perform_pair_wise(matrix1, matrix2, function_):
        result = []
        for i in range(len(matrix1)):
            result.append([])
            for j in range(len(matrix1[0])):
                result[i].append(function_(matrix1[i][j], matrix2[i][j]))
        return np.array(result)

    def plot_distance_function(self, distance_type="unbiased_distance", levels=None):
        if distance_type == "unbiased_distance":
            Z = self.get_unbiased_distance()
        elif distance_type == "weights":
            Z = self.get_weights()
        elif distance_type == "black_box_distance":
            Z = self.get_black_box_distance()
        elif distance_type == "biased_distance":
            Z = self.get_biased_distance()
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

        self.amount_of_points = (self.xmax-self.xmin)*10 + (self.ymax-self.ymin)*10
        self.amount_per_unit = 2
        self.x_spacing = np.linspace(self.xmin, self.xmax, (self.xmax - self.xmin) * self.amount_per_unit)
        self.y_spacing = np.linspace(self.ymin, self.ymax, (self.ymax - self.ymin) * self.amount_per_unit)

        self.points = self.sample_points()
        self.labels = self.get_labels()

    def get_data(self):
        return Data(self.points, self.labels, ["x", "y"])

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

    def plot_setting(self):
        self.plot_black_box_curve()
        self.plot_training_points()

    def plot_local_model(self, test_point):
        #self.plot_points()
        self.plot_black_box_curve()

        leafage = LeafageBinary(self.get_data(), self.labels, random_state, neighbourhood_sampling_strategy="lime")
        local_model = leafage.explain(test_point, self.get_label(test_point)).local_model

        plot_local_model = PlotLocalModel(self.x_spacing, self.y_spacing, local_model)
        self.plot_point(test_point)
        self.plot_points(local_model.neighbourhood.instances, local_model.neighbourhood.labels)
        plot_local_model.plot_distance_function("weights")
        self.plot_training_points()
        #plot_local_model.plot_local_model()

        plt.xlim([self.xmin, self.xmax])
        plt.ylim([self.ymin, self.ymax])

    def test_evaluation(self):
        #train, test, labels_train, labels_test = train_test_split(self.points, self.labels, train_size=0.5)
        leafage = LeafageBinary(self.get_data(), self.labels, random_state)
        evaluation = Faithfulness(self.points, self.labels, leafage.get_local_model, np.arange(0.36, 1, 0.05))

        evaluation.evaluate(self)

    def vary_distance_to_boundary(self):
        #Get the local linear boundary

        leafage = LeafageBinary(self.get_data(), self.labels, random_state)
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

        leafage = LeafageBinary(self.get_data(), self.labels, random_state)
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

# In leafage no scaling


def simple_decision_boundary():
    import warnings
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    random_state = 16
    X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=2, random_state=random_state)
    X = PreProcess(X).transform(X, scale=True)
    data = Data(X, y, ["x", "y"])

    xmin, xmax = X[:, 0].min() - 1, X[:, 0].max() + 1
    ymin, ymax = X[:, 1].min() - 1, X[:, 1].max() + 1

    x_spacing = np.arange(xmin, xmax, 0.1)
    y_spacing = np.arange(ymin, ymax, 0.1)
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    xx, yy = np.meshgrid(x_spacing, y_spacing)

    #classifier = train("knn", X, y, {"n_neighbors": 1})
    classifier = train("nb_g", X, y, {}) # First one
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Y_predicted = classifier.predict(X)
    Z = Z.reshape(xx.shape)

    #plt.contourf(xx, yy, Z, alpha=0.4)
    plt.contour(xx, yy, Z, alpha=0.4)

    #plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y_predicted,
    #            s=20, edgecolor='k')

    leafage = LeafageBinary(data, Y_predicted, random_state,
                            neighbourhood_sampling_strategy="closest_boundary")

    test_point = np.array([-1, 2]) # First one
    #test_point = np.array([-1, 2])
    plt.plot(test_point[0], test_point[1], "r^")
    leafage_linear_model = leafage.get_local_model(test_point, classifier.predict([test_point])[0])
    lime_linear_model = WrapperLime(data, classifier.predict_proba).get_local_model(test_point, None)

    linear_model = leafage_linear_model

    line = Line(linear_model.coefficients[0],
                linear_model.coefficients[1],
                linear_model.original_intercept)

    y = [line(t) for t in np.arange(xmin, xmax, 0.1)]
    plt.plot(x_spacing, y)

    #plt.title('Value of a house')
    #plt.ylabel('Area')
    #plt.xlabel('Age')
    plt.xticks(range(-3,5), range(1, 9))
    plt.yticks(range(-3,4), range(30, 37))

    plt.show()


def complex_decision_boundary():
    test_point = np.array([1, 2])
    random_state = 20
    random.seed(random_state)

    import warnings
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    plt.figure()

    X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=2, random_state=random_state)
    xmin, xmax = -4, 5
    ymin, ymax = -4, 5

    classifier = train("knn", X, y, {"n_neighbors": 10})


    extra_points = []
    for _ in range(15000):
        extra_points.append([xmin + (xmax-xmin)*random.random(), ymin + (ymax-ymin)*random.random()])

    extra_points_prediction = classifier.predict(extra_points)

    #X = np.concatenate((X, np.array(extra_points)))
    #y = np.concatenate((y, extra_points_prediction), axis=None)
    X = np.array(extra_points)
    y = extra_points_prediction

    data = Data(X, y, ["x", "y"])

    #x_spacing = np.arange(xmin, xmax, 0.1)
    #y_spacing = np.arange(ymin, ymax, 0.1)
    x_spacing = np.linspace(xmin, xmax, (xmax - xmin) * 30)
    y_spacing = np.linspace(ymin, ymax, (ymax - ymin) * 30)

    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    xx, yy = np.meshgrid(x_spacing, y_spacing)

    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Y_predicted = classifier.predict(X)
    Z = Z.reshape(xx.shape)

    #plt.contourf(xx, yy, Z, alpha=0.4)
    plt.contour(xx, yy, Z, alpha=0.4, level=[0])

    #plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y_predicted,
    #            s=20, edgecolor='k')

    leafage = LeafageBinary(data, Y_predicted, random_state,
                            neighbourhood_sampling_strategy="closest_boundary")

    plt.plot(test_point[0], test_point[1], "r^")
    leafage_local_model = leafage.explain(test_point, classifier.predict([test_point])[0]).local_model
    leafage_linear_model = leafage_local_model.linear_model
    #lime_linear_model = WrapperLime(data, classifier.predict_proba).get_local_model(test_point, None)

    linear_model = leafage_linear_model

    line = Line(linear_model.coefficients[0],
                linear_model.coefficients[1],
                linear_model.original_intercept)

    y = [line(t) for t in x_spacing]
    #plt.plot(x_spacing, y)

    #plot_local_model = PlotLocalModel(x_spacing, y_spacing, leafage_local_model)
    #plot_local_model.plot_distance_function("final_distance")
    Z2 = np.array([leafage_local_model.distances.get_final_distance(x) for x in np.c_[xx.ravel(), yy.ravel()]])
    Z2 = Z2.reshape(xx.shape)

    plt.contourf(xx, yy, Z2, 20, cmap='RdGy', levels=np.arange(0,7,0.3))
    plt.colorbar()

    plt.xticks(range(-4, 6), range(1, 11))
    plt.yticks(range(-4, 6), range(30, 41))


def linear_approximation():
    test_point = np.array([1, 2])
    random_state = 20
    random.seed(random_state)

    import warnings
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    plt.figure()

    X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=2, random_state=random_state)
    xmin, xmax = -4, 5
    ymin, ymax = -4, 5

    classifier = train("knn", X, y, {"n_neighbors": 10})

    extra_points = []
    for _ in range(15000):
        extra_points.append([xmin + (xmax-xmin)*random.random(), ymin + (ymax-ymin)*random.random()])

    extra_points_prediction = classifier.predict(extra_points)

    X = np.array(extra_points)
    y = extra_points_prediction

    data = Data(X, y, ["x", "y"])

    x_spacing = np.linspace(xmin, xmax, (xmax - xmin) * 30)
    y_spacing = np.linspace(ymin, ymax, (ymax - ymin) * 30)

    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    xx, yy = np.meshgrid(x_spacing, y_spacing)

    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Y_predicted = classifier.predict(X)
    Z = Z.reshape(xx.shape)

    #plt.contourf(xx, yy, Z, alpha=0.4)
    plt.contour(xx, yy, Z, alpha=0.4, level=[0])

    #plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y_predicted,
    #            s=20, edgecolor='k')

    leafage = LeafageBinary(data, Y_predicted, random_state,
                            neighbourhood_sampling_strategy="closest_boundary")

    plt.plot(test_point[0], test_point[1], "r^")
    leafage_local_model = leafage.explain(test_point, classifier.predict([test_point])[0]).local_model
    leafage_linear_model = leafage_local_model.linear_model
    #lime_linear_model = WrapperLime(data, classifier.predict_proba).get_local_model(test_point, None)

    linear_model = leafage_linear_model

    #plot_local_model = PlotLocalModel(x_spacing, y_spacing, leafage_local_model)
    #plot_local_model.plot_distance_function("final_distance")
    Z2 = np.array([leafage_local_model.distances.get_final_distance(x) for x in np.c_[xx.ravel(), yy.ravel()]])
    Z2 = Z2.reshape(xx.shape)

    plt.contourf(xx, yy, Z2, 20, cmap='RdGy', levels=np.arange(0,14,0.6))
    plt.colorbar()

    line = Line(linear_model.coefficients[0],
                linear_model.coefficients[1],
                linear_model.original_intercept)

    y = [line(t) for t in x_spacing]
    plt.plot(x_spacing, y)

    plt.xticks(range(-4, 6), range(1, 11))
    plt.yticks(range(-4, 6), range(30, 41))


if __name__ == "__main__":
    #complex_decision_boundary()
    #linear_approximation()

    a = TwoDimensionExample()
    a.plot_local_model(np.array([-8,4]))
    plt.show()




