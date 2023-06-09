import csv
import json
from pathlib import Path
from typing import List, Dict
import logging

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pydantic import BaseModel, Field, validator
import itertools

# set up logging for info and above
logging.basicConfig(level=logging.CRITICAL)


class SUBDATA(BaseModel):
    P: int
    SR: float
    cP: float
    T: float
    SS: float

    # check the P value is between 0 and 100 using a validator
    @validator("P")
    def check_P(cls, v):
        if 0 <= v <= 100:
            logging.info(f"Valid P value {v}")
        else:
            logging.warning(f"Invalid P value {v}")
        return v


class Data(BaseModel):
    HPyBF4_CP: List[SUBDATA]
    HPyBF4_CT: List[SUBDATA]
    HPyBr_CP: List[SUBDATA]
    HPyBr_CT: List[SUBDATA]
    OPyBr_CP: List[SUBDATA]
    BMIMBr_CP_CT: List[SUBDATA]
    BMIMBr_CT: List[SUBDATA]
    DES117_CP: List[SUBDATA]


class MyWork:
    def __init__(self, *args, **kwargs):
        self.data = None

    def make_heatmap(self, my_dict, ILs, temperature, pressure):
        df = pd.DataFrame(my_dict[f"{ILs}_t_{temperature}_p_{pressure}"])
        # Create a pivot table to rearrange the data
        df_pivot = df.pivot_table(values="cP", index="SS", columns="SR")
        # Create the heat map
        try:
            sns.heatmap(df, cmap="coolwarm", annot=True,
                        fmt=".2f", cbar_kws={"label": "cP"}
                        )
            plt.xlabel("SR")
            plt.ylabel("SS")
            plt.title("Heat Map: SS vs SR with cP")
            plt.show()
        except ValueError:
            print(f"No data for {ILs}_t_{temperature}_p_{pressure}")

    def method_1(self):
        temperatures = [74, 120, 220, 310, 410, 480]
        pressures = [100, 500, 800, 1300, 1800]
        pairs = list(itertools.product(temperatures, pressures))
        new_data = []
        for ILs in data_model.dict().keys():
            print(ILs)
            my_dict = {}

            ILs_data = data_model.__dict__[ILs]
            for pair in pairs:
                temperature, pressure = pair
                my_dict[f"{ILs}_t_{temperature}_p_{pressure}"] = {
                    "SR": [x.SR for x in ILs_data if x.T == temperature and x.P == pressure],
                    "SS": [x.SS for x in ILs_data if x.T == temperature and x.P == pressure],
                    "cP": [x.cP for x in ILs_data if x.T == temperature and x.P == pressure],
                }

                self.make_heatmap(my_dict, ILs, temperature, pressure)

            new_data.append(my_dict)
        return new_data


if __name__ == "__main__":
    # read the data json file from the data folder
    data_folder = Path("data/")
    file_to_open = data_folder / "data.json"
    with open(file_to_open) as json_file:
        data = json.load(json_file)

    # print(data)

    # parse the data into a pydantic model
    data_model = Data(**data)

    obj = MyWork(data_model)
    obj.method_1()