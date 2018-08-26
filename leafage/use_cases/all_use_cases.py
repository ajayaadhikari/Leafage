from handwritten_digits import DigitsDataset
from iris import IrisDataSet
from adult import Adult
from housing import HousingDataSet
from breast_cancer import BreastCancerDataset
from wine import WineDataset

all_data_sets = {"iris": IrisDataSet,
                 "digits": DigitsDataset,
                 "adult": Adult,
                 "housing": HousingDataSet,
                 "wine": WineDataset,
                 "breast_cancer": BreastCancerDataset}

if __name__ == "__main__":
    a = WineDataset()
    b = BreastCancerDataset()