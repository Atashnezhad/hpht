import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import List


class MyPlot:
    def __init__(self):
        self.ils_names: List = []

    def filter_data(self, data, il_name):
        filtered_data = pd.DataFrame(data[il_name])
        return filtered_data

    def group_data(self, filtered_data, group_by_parameter: str = "T"):
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
    ):
        self.ils_names = list(data.keys())

        filtered_data = self.filter_data(data, il_name)
        list_unique = self.group_data(filtered_data, grouop_by_parameter)

        # fig size
        plt.figure(figsize=(12, 8))
        for case in list_unique:
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
    ):
        self.ils_names = list(data.keys())

        filtered_data = self.filter_data(data, il_name)
        list_unique = self.group_data(filtered_data, grouop_by_parameter)

        # fig size
        plt.figure(figsize=(12, 8))
        for case in list_unique:
            # separate data by parameter
            data = filtered_data[filtered_data[grouop_by_parameter] == case]
            plt.plot(
                data[x_parameter],
                data[y_parameter],
                label=f"{grouop_by_parameter} = {case} {legend_units}",
                marker="o",
                linestyle="--",
            )

            # fit a line to the data
            m, b = np.polyfit(data[x_parameter], data[y_parameter], 1)
            plt.plot(
                data[x_parameter],
                m * data[x_parameter] + b,
                label=f"y = {m:.2f}x + {b:.2f}",
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
        plt.show()
