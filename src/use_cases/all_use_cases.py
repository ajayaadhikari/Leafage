from handwritten_digits import DigitsDataset
from iris import IrisDataSet
from adult import Adult
from src.use_cases.housing import HousingDataSet

different_data_sets = {"iris": IrisDataSet,
                       "digits": DigitsDataset,
                       "adult": Adult,
                       "housing": HousingDataSet}

if __name__ == "__main__":
    a = HousingDataSet()
    b = 8