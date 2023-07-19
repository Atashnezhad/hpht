from pathlib import Path
import os
import sys
import pandas as pd


if __name__ == "__main__":
    # read the data from the data_nick dir
    data_dir = Path(__file__).parent / "data_nick"
    # read the csv files
    data_15RAW = pd.read_csv(data_dir / "15RAW.csv")
    data_6RAW = pd.read_csv(data_dir / "6RAW.csv")
    print(data_15RAW.head())
    print(data_6RAW.head())
