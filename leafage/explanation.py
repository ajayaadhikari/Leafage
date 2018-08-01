import pandas as pd
class Explanation:
    def __init__(self,
                 test_instance,
                 examples_in_support,
                 examples_against,
                 local_model,
                 fact_class,
                 foil_class,
                 feature_names):

        self.test_instance = pd.Series(test_instance, index=feature_names)
        self.examples_in_support = pd.DataFrame(examples_in_support, columns=feature_names)
        self.examples_against = pd.DataFrame(examples_against, columns=feature_names)
        self.local_model = local_model
        self.fact_class = fact_class
        self.foil_class = foil_class
        self.feature_names = feature_names
