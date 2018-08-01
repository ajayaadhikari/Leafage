
class Explanation:
    def __init__(self,
                 test_instance,
                 examples_in_support,
                 examples_against,
                 local_model,
                 fact_class,
                 foil_class):

        self.test_instance = test_instance
        self.examples_in_support = examples_in_support
        self.examples_against = examples_against
        self.local_model = local_model
        self.fact_class = fact_class
        self.foil_class = foil_class
