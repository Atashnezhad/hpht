import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from typing import List


class MyPlot:
    def __init__(self):
        self.ils_names: List = []
        self.correlation_coefficient: List = []

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
        y_axis_lim: tuple = None,
        x_axis_lim: tuple = None,
    ):
        self.ils_names = list(data.keys())

        filtered_data = self.filter_data(data, il_name)
        list_unique = self.group_data(filtered_data, grouop_by_parameter)

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
            list_unique = self.group_data(filtered_data, grouop_by_parameter)
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
                label=f"{grouop_by_parameter}={case}{legend_units}\nSS={p[1]:.2f}*SR+{p[0]:.2f}, R2={np.corrcoef(data[x_parameter], data[y_parameter])[0,1]**2:.2f}",
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
            list_unique = self.group_data(filtered_data, grouop_by_parameter)
        list_unique = temps_used_fit

        # fig size
        plt.figure(figsize=(12, 8))
        for case in list_unique:
            # font size
            plt.rcParams.update({"font.size": font_size})
            # separate data by parameter
            data = filtered_data[filtered_data[grouop_by_parameter] == case]
            # plt.scatter(
            #     data[x_parameter],
            #     data[y_parameter],
            #     # label=f"{group_by_parameter} = {case} {legend_units}",
            #     marker="o",
            # )

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

            # plt.plot(
            #     data[x_parameter],
            #     np.exp(p(np.log(data[x_parameter]))),
            #     linestyle="--",
            #     # color="black",
            #     label=f"{grouop_by_parameter}={case}{legend_units}\nSS={np.exp(p[1]):.2f}*SR^{p[0]:.2f}, R2={np.corrcoef(np.log(data[x_parameter]), np.log(data[y_parameter]))[0,1]**2:.2f}",
            # )

            # legend in the top right corner
            # plt.legend(loc="upper right")
            # plt.xlabel(x_axis_label)
            # plt.ylabel(y_axis_label)
            # # font size
            # plt.rcParams.update({"font.size": font_size})
            # # axis font size
            # plt.tick_params(axis="both", which="major", labelsize=axis_font_size)
            # if y_axis_log_scale:
            #     # y axis log scale
            #     plt.yscale("log")
            # plt.title(maping_dict[il_name])
            # # make small grid
            # plt.grid(True, which="both", ls="--", color="0.65")
            #
            # # change the figure background color
            # plt.gca().set_facecolor("w")
            #
            # if y_axis_lim:
            #     plt.ylim(y_axis_lim)
            # if x_axis_lim:
            #     plt.xlim(x_axis_lim)

        plt.show()

    def power_law(self, x, k, n):
        return k * np.power(x, n)
