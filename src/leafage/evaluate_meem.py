from src.leafage.meem import SetupExplanatoryExamplesMeem
from src.utils.stopwatch import stopwatch

stopwatch.turn_off_print()
stopwatch.start()

from src.leafage.explanatory_examples import SetupVariables
#setup = SetupVariables("adult", 0.6, 11, "svc", "leafage", {"kernel": "linear", "probability": True}, {})
#setup = SetupVariables("adult", 0.6, 11, "rf", "leafage", {}, {})
setup = SetupVariables("iris", 0.7, 11, "lr")

c = SetupExplanatoryExamplesMeem(setup)
# c.evaluation.plot()
# c.explanatory_examples.visualize(c.test[0])
