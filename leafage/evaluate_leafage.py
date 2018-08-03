import warnings

from leafage import Leafage
from scenario import Scenario

warnings.filterwarnings(action='ignore', category=DeprecationWarning)


def housing_from_use_cases():
    #setup = SetupVariables("adult", 0.6, 11, "svc", "leafage", {"kernel": "linear", "probability": True}, {})
    #setup = SetupVariables("adult", 0.6, 11, "rf", "leafage", {}, {})
    scenario = Scenario("load_from_use_cases", "housing", "lr")
    leafage = scenario.leafage
    explanation = scenario.get_explanation(leafage.training_data.feature_vector[0], 5)
    explanation.visualize_feature_importance(target="write_to_file", path="../output/feature_importance.png")
    explanation.visualize_examples(target="write_to_file", path="../output/examples_in_support.png", type="examples_in_support")
    explanation.visualize_examples(target="write_to_file", path="../output/examples_against.png", type="examples_against")


    # c.evaluation.plot()
    # c.explanatory_examples.visualize(c.test[0])


def housing_from_file():
    scenario = Scenario("load_from_file", "../data/housing/pre_processed_train.csv", "lr")
    leafage = scenario.leafage
    explanation = scenario.get_explanation(scenario.data.feature_vector[0], amount_of_examples=5)
    a = 4


housing_from_use_cases()
