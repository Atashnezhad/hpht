import json
import logging
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from typing import List, Dict, Union, Tuple
from utils import handle_default_values


class MyPlot:
    def __init__(self):
        self.ils_names: List = []
        self.correlation_coefficient: List = []

    def filter_data(self, data, il_name):
        filtered_data = pd.DataFrame(data[il_name])
        return filtered_data

    def list_of_unique(self, filtered_data, group_by_parameter: str = "T"):
        list_unique = filtered_data[group_by_parameter].unique()
        return list_unique

    def plot_data(
            self,
            data,
            il_name,
            grouop_by_parameter: str = "T",
            x_parameter: str = "SR",
            y_parameter: str = "SS",
            y_axis_log_scale: bool = False,
            legend_units: str = "F",
            x_axis_label: str = "SR",
            y_axis_label: str = "SS",
            maping_dict: dict = None,
            font_size: int = 18,
            axis_font_size: int = 18,
            y_axis_lim: tuple = None,
            x_axis_lim: tuple = None,
    ):
        self.ils_names = list(data.keys())

        filtered_data = self.filter_data(data, il_name)
        list_unique = self.list_of_unique(filtered_data, grouop_by_parameter)

        # fig size
        plt.figure(figsize=(12, 8))
        for case in list_unique:
            # font size
            plt.rcParams.update({"font.size": font_size})
            # separate data by parameter
            data = filtered_data[filtered_data[grouop_by_parameter] == case]
            plt.plot(
                data[x_parameter],
                data[y_parameter],
                label=f"{grouop_by_parameter} = {case} {legend_units}",
                marker="o",
                linestyle="--",
            )
            # legend in the top right corner
            plt.legend(loc="upper right")
            plt.xlabel(x_axis_label)
            plt.ylabel(y_axis_label)

            # y axis limit
            if y_axis_lim:
                plt.ylim(y_axis_lim)

            # axis font size
            plt.tick_params(axis="both", which="major", labelsize=axis_font_size)
            if y_axis_log_scale:
                # y axis log scale
                plt.yscale("log")
            plt.title(maping_dict[il_name])
            # make small grid
            plt.grid(True, which="both", ls="--", color="0.65")

            # change the figure background color
            plt.gca().set_facecolor("w")
        plt.show()

    def plot_data_2(
            self,
            data,
            il_name,
            grouop_by_parameter: str = "T",
            x_parameter: str = "SR",
            y_parameter: str = "SS",
            y_axis_log_scale: bool = False,
            legend_units: str = "F",
            x_axis_label: str = "SR",
            y_axis_label: str = "SS",
            maping_dict: dict = None,
            font_size: int = 18,
            axis_font_size: int = 18,
            temps_used_fit: list = None,
            y_axis_lim: tuple = None,
            x_axis_lim: tuple = None,
    ):
        self.ils_names = list(data.keys())

        filtered_data = self.filter_data(data, il_name)
        if not temps_used_fit:
            list_unique = self.list_of_unique(filtered_data, grouop_by_parameter)
        list_unique = temps_used_fit

        # fig size
        plt.figure(figsize=(12, 8))
        for case in list_unique:
            # font size
            plt.rcParams.update({"font.size": font_size})
            # separate data by parameter
            data = filtered_data[filtered_data[grouop_by_parameter] == case]
            plt.scatter(
                data[x_parameter],
                data[y_parameter],
                # label=f"{group_by_parameter} = {case} {legend_units}",
                marker="o",
            )

            # fit a line to the data and show the fit equation and R^2 in the legend and use the same color as the data
            z = np.polyfit(data[x_parameter], data[y_parameter], 1)
            p = np.poly1d(z)

            # save the z and p and r2 in the self.correlation_coefficient
            if grouop_by_parameter == "T":
                my_lable = "Temperature (F)"
            elif grouop_by_parameter == "P":
                my_lable = "Pressure (psi)"
            else:
                my_lable = "my_lable"

            self.correlation_coefficient.append(
                {
                    "Ionic Liquid": maping_dict[il_name],
                    my_lable: round(case, 2),
                    # "z": z,
                    "Slope": round(p[1], 3),
                    "Intercept": round(p[0], 3),
                    "r2": round(np.corrcoef(data[x_parameter], data[y_parameter])[0, 1] ** 2, 2),
                }
            )

            plt.plot(
                data[x_parameter],
                p(data[x_parameter]),
                linestyle="--",
                # color="black",
                label=f"{grouop_by_parameter}={case}{legend_units}\nSS={p[1]:.2f}*SR+{p[0]:.2f}, R2={np.corrcoef(data[x_parameter], data[y_parameter])[0, 1] ** 2:.2f}",
            )

            # legend in the top right corner
            plt.legend(loc="upper right")
            plt.xlabel(x_axis_label)
            plt.ylabel(y_axis_label)
            # font size
            plt.rcParams.update({"font.size": font_size})
            # axis font size
            plt.tick_params(axis="both", which="major", labelsize=axis_font_size)
            if y_axis_log_scale:
                # y axis log scale
                plt.yscale("log")
            plt.title(maping_dict[il_name])
            # make small grid
            plt.grid(True, which="both", ls="--", color="0.65")

            # change the figure background color
            plt.gca().set_facecolor("w")

            if y_axis_lim:
                plt.ylim(y_axis_lim)
            if x_axis_lim:
                plt.xlim(x_axis_lim)

        plt.show()

    def plot_data_3(
            self,
            data,
            il_name,
            grouop_by_parameter: str = "T",
            x_parameter: str = "SR",
            y_parameter: str = "SS",
            y_axis_log_scale: bool = False,
            legend_units: str = "F",
            x_axis_label: str = "SR",
            y_axis_label: str = "SS",
            maping_dict: dict = None,
            font_size: int = 18,
            axis_font_size: int = 18,
            temps_used_fit: list = None,
            y_axis_lim: tuple = None,
            x_axis_lim: tuple = None,
    ):
        self.ils_names = list(data.keys())

        filtered_data = self.filter_data(data, il_name)
        if not temps_used_fit:
            list_unique = self.list_of_unique(filtered_data, grouop_by_parameter)
        list_unique = temps_used_fit

        # fig size
        plt.figure(figsize=(12, 8))
        for case in list_unique:
            # font size
            plt.rcParams.update({"font.size": font_size})
            # separate data by parameter
            data = filtered_data[filtered_data[grouop_by_parameter] == case]

            # fit a power law to the data
            # Generate some example data
            # print("data[x_parameter]\n", data[x_parameter])
            # print("np.array(data[y_parameter])\n", np.array(data[y_parameter]))
            x_data = np.array(data[x_parameter])
            y_data = np.array(data[y_parameter])

            # Fit the power law model to the data
            popt, pcov = curve_fit(self.power_law, x_data, y_data)
            # Extract the optimized parameters
            k_opt, n_opt = popt

            # Calculate the predicted values using the fitted model
            y_pred = self.power_law(x_data, k_opt, n_opt)

            # Calculate the residuals
            residuals = y_data - y_pred

            # Calculate the total sum of squares (TSS) and residual sum of squares (RSS)
            TSS = np.sum((y_data - np.mean(y_data)) ** 2)
            RSS = np.sum(residuals ** 2)

            # Calculate the R-squared value
            R_squared = 1 - (RSS / TSS)

            # save the z and p and r2 in the self.correlation_coefficient
            if grouop_by_parameter == "T":
                my_lable = "Temperature (F)"
            elif grouop_by_parameter == "P":
                my_lable = "Pressure (psi)"
            else:
                my_lable = "my_lable"

            self.correlation_coefficient.append(
                {
                    "Ionic Liquid": maping_dict[il_name],
                    my_lable: round(case, 2),
                    "n": round(n_opt, 3),
                    "K": round(k_opt, 3),
                    "r2": round(R_squared, 2),
                }
            )

        plt.show()

    def power_law(self, x, k, n):
        return k * np.power(x, n)

    def get_x_y_params(
            self,
            data: Dict[str, List[Dict[str, Union[float, int]]]],
            il_name: str = "HPyBF4_CP",
            grouop_by_parameter: Tuple = ("T", 74),
            x_parameter: str = "SR",
            y_parameter: str = "SS",
    ):
        il_data = self.filter_data(data, il_name)
        # get the data where the T is 75
        # il_data = pd.DataFrame(il_data)
        il_data = il_data[il_data[grouop_by_parameter[0]] == grouop_by_parameter[1]]
        x_param = il_data[x_parameter]
        y_param = il_data[y_parameter]
        return x_param, y_param

    @handle_default_values
    def plot_data_4(
            self,
            data: Dict[str, List[Dict[str, Union[float, int]]]],
            il_names: List[str] = None,
            grouop_by_parameter: Tuple = ("T", 74),
            x_parameter: str = "SR",
            y_parameter: str = "SS",
            font_size: int = 18,
            maping_dict: dict = None,
            title: str = "title",
            y_lim: tuple = None,
            x_label: str = "x_label",
            y_label: str = "y_label",
    ):
        self.ils_names = il_names

        my_dict = defaultdict(dict)
        for il_name in self.ils_names:
            x_param, y_param = self.get_x_y_params(data,
                                                   il_name,
                                                   x_parameter=x_parameter,
                                                   y_parameter=y_parameter,
                                                   grouop_by_parameter=(grouop_by_parameter)
                                                   )
            my_dict[il_name][x_parameter] = x_param
            my_dict[il_name][y_parameter] = y_param

        plt.figure(figsize=(12, 8))
        # font size
        plt.rcParams.update({"font.size": font_size})
        # for all the ionic liquids in the my_dict
        # plot the data
        for il_name in my_dict.keys():
            plt.plot(
                my_dict[il_name][x_parameter],
                my_dict[il_name][y_parameter],
                label=maping_dict[il_name],
                marker="o",
                linestyle="--",
            )
            # x axis label
            plt.xlabel(x_label)
            # y axis label
            plt.ylabel(y_label)

        # make small grid
        plt.grid(True, which="both", ls="--", color="0.65")
        plt.title(title+" "+str(grouop_by_parameter[0])+" = "+str(grouop_by_parameter[1]))
        # use y_lim to set the y_axis limit
        if y_lim:
            plt.ylim(y_lim)
        # change the figure background color
        plt.gca().set_facecolor("w")
        plt.legend()
        plt.show()


if __name__ == "__main__":

    mapping_dict = {
        "HPyBF4_CP": "1-Hexylpyridinium tetrafluoroborate (HPyBF4)",
        "HPyBr_CP": "N-Hexylpyridinium bromide (HPyBr)",
        "OPyBr_CP": "1-Octylpyridinium bromide (OPyBr)",
        "HPyBF4_CT": "1-Hexylpyridinium tetrafluoroborate (HPyBF4)",
        "HPyBr_CT": "N-Hexylpyridinium bromide (HPyBr)",
        "BMIMBr_CP_CT": "1-Butyl-3-methylimidazolium bromide (BMIMBr)",
        "BMIMBr_CT": "1-Butyl-3-methylimidazolium bromide (BMIMBr)",
        "DES117_CP": "ChCl-MgCl2(6H2O) (DES117)"
    }

    # read data from the data folder
    data_folder = Path("__file__").parent / "data"
    data_file_name = "data.json"
    with open(data_folder / data_file_name, "r") as f:
        json_data = f.read()
    # Parse the JSON data
    data = json.loads(json_data)

    obj = MyPlot()
    my_dict = obj.plot_data_4(
        data,

        il_names=["HPyBF4_CP", "HPyBr_CP", "OPyBr_CP"],
        grouop_by_parameter=("T", 74),

        # il_names=["HPyBF4_CT", "HPyBr_CT"],
        # grouop_by_parameter=("P", 1800),

        font_size=25,
        maping_dict=mapping_dict,
        title="constant temperature",
        y_lim=(0, 500),
        x_label="Shear rate (1/sec)",
        y_label="Shear stress (dyne/cm^2)",
    )
