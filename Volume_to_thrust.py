import pandas as pd
import numpy as np
import re
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from methods import *

f_path = "/Users/beelee/PycharmProjects/OblateThrust/csv/"
a_digitale_f = f_path + "a_digitale_data.csv"
s_sp_f = f_path + "s_sp_data.csv"
p_flavicirrata_f = f_path + "p_flavicirrata_data.csv"
a_victoria_f = f_path + "a_victoria_data.csv"
m_cellularia_f = f_path + "m_cellularia_data.csv"
p_gregarium_f = f_path + "p_gregarium_data.csv"

sea_den = 1.024 * np.power(10.0, 6)  # g/m^3, 1.024 g/cm^3 (Colin & Costello, 2001)
sea_vis = np.power(10.0, -6)  # m^2/s


######################################################################
# TASK 4:
# to build MODEL3 that gives a more reliable estimates on the maximum
# thrust based on the wake volume, which we suppose is directly correlated to
# 3 times the volume change of an oblate bell. MODEL3 is still inspired by
# MODEL1
######################################################################
def main():

    # import pre-cleaned up csv files, 1 prolate 1 oblate
    # import pre-cleaned up csv files, 3 oblates
    a_digitale = pd.read_csv(a_digitale_f)
    s_sp = pd.read_csv(s_sp_f)
    p_flavicirrata = pd.read_csv(p_flavicirrata_f)
    # import pre-cleaned up csv files, 3 oblates
    a_victoria = pd.read_csv(a_victoria_f)
    m_cellularia = pd.read_csv(m_cellularia_f)
    p_gregarium = pd.read_csv(p_gregarium_f)
    # group the medusa of interest for easy access
    dfs = [a_digitale, s_sp, p_flavicirrata, a_victoria, m_cellularia, p_gregarium]
    name = ["a_digitale", "s_sp", "p_flavicirrata", "a_victoria", "m_cellularia", "p_gregarium"]

    for df in dfs:
        clean_time(df)
        get_basics(df)
        get_thrust(df)

    dfss = [split_digitale(dfs[0]), split_sp(dfs[1]), split_flavicirrata(dfs[2]),
            split_victoria(dfs[3]), split_cellularia(dfs[4]), split_gregarium(dfs[5])]

    dfs_count = 0
    for dfs in dfss:
        total_x = np.array([])
        total_y = np.array([])
        colors = ['goldenrod', 'firebrick', 'forestgreen', 'dodgerblue']
        df_count = 0
        for df in dfs:
            label = "cycle #" + str(df_count + 1)
            plt.plot(df["st"], df["V"], color=colors[df_count], label=label)
            total_x = np.append(total_x, df["st"].values)
            total_y = np.append(total_y, df["V"].values)
            df_count += 1
        input_regression = [('polynomial', PolynomialFeatures(degree=2)), ('modal', LinearRegression())]
        pipe = Pipeline(input_regression)
        pipe.fit(total_x.reshape(-1, 1), total_y.reshape(-1, 1))
        poly_pred = pipe.predict(total_x.reshape(-1, 1))
        sorted_zip = sorted(zip(total_x, poly_pred))
        x_poly, poly_pred = zip(*sorted_zip)
        reg_label = ('RMSE %f' % np.sqrt(mean_squared_error(x_poly, poly_pred)))
        plt.plot(x_poly, poly_pred, color='magenta', linestyle='--', label=reg_label)
        plt.title("%s bell volume over 1 pulsation cycle" % (name[dfs_count]))
        plt.xlabel("percentage cycle %")
        plt.ylabel("volume m^3")
        plt.tight_layout()
        plt.show()
        dfs_count += 1

    dfs_count = 0
    total_x1 = np.array([])
    total_y1 = np.array([])
    total_x2 = np.array([])
    total_y2 = np.array([])
    for dfs in dfss:
        for df in dfs:
            bel_vol = df["V"].max() - df["V"].min()
            if dfs_count <= 2:
                ej_vol = dS(bel_vol, False)
                total_x1 = np.append(total_x1, ej_vol)
                total_y1 = np.append(total_y1, df["tf"].max())
            else:
                ej_vol = dS(bel_vol, True)
                total_x2 = np.append(total_x2, ej_vol)
                total_y2 = np.append(total_y2, df["tf"].max())
        dfs_count += 1
    plt.scatter(total_x1, total_y1)
    plt.title("effect of ejected fluid volume on prolate max thrust")
    plt.xlabel("ejected volume")
    plt.ylabel("maximum thrust g*m*s^-2")
    plt.tight_layout()
    plt.show()

    plt.scatter(total_x2, total_y2)
    plt.title("effect of ejected fluid volume on oblate max thrust")
    plt.xlabel("ejected volume")
    plt.ylabel("maximum thrust g*m*s^-2")
    plt.tight_layout()
    plt.show()


main()
