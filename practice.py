import csv
import json
from pathlib import Path
from typing import List, Dict
import logging
from pydantic import BaseModel, Field, validator
from collections import defaultdict
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


if __name__ == "__main__":
    # read the data json file from the data folder
    data_folder = Path("data/")
    file_to_open = data_folder / "data.json"
    with open(file_to_open) as json_file:
        data = json.load(json_file)

    # print(data)
    # parse the data into a pydantic model
    data_model = Data(**data)

    HPyBF4_CP = data_model.HPyBF4_CP
    hpy_bf3_cp_t_74 = {
        "SR": [x.SR for x in HPyBF4_CP if x.T == 74],
        "SS": [x.SS for x in HPyBF4_CP if x.T == 74]
    }

    hpy_bf3_cp_t_120 = {
        "SR": [x.SR for x in HPyBF4_CP if x.T == 120],
        "SS": [x.SS for x in HPyBF4_CP if x.T == 120]
    }

    hpy_bf3_cp_t_220 = {
        "SR": [x.SR for x in HPyBF4_CP if x.T == 220],
        "SS": [x.SS for x in HPyBF4_CP if x.T == 220]
    }

    hpy_bf3_cp_t_310 = {
        "SR": [x.SR for x in HPyBF4_CP if x.T == 310],
        "SS": [x.SS for x in HPyBF4_CP if x.T == 310]
    }

    hpy_bf3_cp_t_410 = {
        "SR": [x.SR for x in HPyBF4_CP if x.T == 410],
        "SS": [x.SS for x in HPyBF4_CP if x.T == 410]
    }

    hpy_bf3_cp_t_480 = {
        "SR": [x.SR for x in HPyBF4_CP if x.T == 480],
        "SS": [x.SS for x in HPyBF4_CP if x.T == 480]
    }
