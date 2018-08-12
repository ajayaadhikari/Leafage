from handwritten_digits import DigitsDataset
from iris import IrisDataSet
from adult import Adult
from housing import HousingDataSet

all_data_sets = {"iris": IrisDataSet,
                 "digits": DigitsDataset,
                 "adult": Adult,
                 "housing": HousingDataSet}

if __name__ == "__main__":
    a = HousingDataSet()
    b = 8