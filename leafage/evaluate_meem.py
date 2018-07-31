import warnings

from meem import SetupExplanatoryExamplesMeem
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

from explanatory_examples import SetupVariables
#setup = SetupVariables("adult", 0.6, 11, "svc", "leafage", {"kernel": "linear", "probability": True}, {})
#setup = SetupVariables("adult", 0.6, 11, "rf", "leafage", {}, {})
setup = SetupVariables("iris", 1, 11, "lr")

c = SetupExplanatoryExamplesMeem(setup)
explanation = c.explain(c.training_data.feature_vector[0])
a = 4
# c.evaluation.plot()
# c.explanatory_examples.visualize(c.test[0])
