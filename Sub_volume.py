import pandas as pd
import numpy as np
import re
from matplotlib import pyplot as plt
from matplotlib.pyplot import figtext
from methods import *

f_path = "/Users/beelee/PycharmProjects/OblateThrust/csv/"
s_sp_f = f_path + "s_sp_data.csv"
p_gregarium_f = f_path + "p_gregarium_data.csv"

sea_den = 1.024 * np.power(10.0, 6)  # g/m^3, 1.024 g/cm^3 (Colin & Costello, 2001)
sea_vis = np.power(10.0, -6)  # m^2/s


######################################################################
# TASK 2:
# to make adjustments to MODEL1 and validate the proposed improvement,
# an extra parameter is required and yet it is not accessible.
# the change in the bell's subumbrellar volume per pulsation is crucial
# for building MODEL2. an auxillary model is constructed to make sound
# estimates of that parameter based the change of bell volume per pulsation
# RESULT:
######################################################################
def main():

    # import pre-cleaned up csv files, 1 prolate 1 oblate
    s_sp = pd.read_csv(s_sp_f)
    p_gregarium = pd.read_csv(p_gregarium_f)
    # group the two medusae of interest for easy access
    dfs = [s_sp, p_gregarium]
    name = ['s.sp', 'p.gregarium']

    # s_sp: 0.85cm diameter, p_gregarium: 2.14cm diameter (Colin & Costello, 2001)
    s_sp_ori = ori(0.85 / 100)
    p_gregarium_ori = ori(2.14 / 100)
    # group the two constant orifice areas for easy access
    oris = [s_sp_ori, p_gregarium_ori]

    for (df, o) in zip(dfs, oris):
        clean_time(df)
        get_basics(df)
        get_thrust(df)
        sub_n_vol_change(df, o)

    dfss = [split_sp(dfs[0]), split_gregarium(dfs[1])]

    dfs_count = 0
    for dfs in dfss:
        total_x = np.array([])
        total_y = np.array([])
        for df in dfs:
            total_x = np.append(total_x, df["dV"].values)
            total_y = np.append(total_y, df["dS"].values)
        plt.scatter(total_x, total_y)
        r = np.corrcoef(total_x, total_y)
        polymodel = np.poly1d(np.polyfit(total_x, total_y, 1))
        m, b = polymodel
        polyline = np.linspace(min(total_x), max(total_x), 100)
        plt.plot(polyline, polymodel(polyline), label='dV to dS regression')
        plt.title("change of %s subumbrellar volume with respect to change of bell volume" % (name[dfs_count]))
        # chisq = sum(np.power((total_y - polymodel2(total_x)), 2)/polymodel2(total_x))
        figtext(0.15, 0.8, "r: %.2f" % r[0, 1])
        figtext(0.15, 0.75, "line: %.3f x + %.2E" % (m, b))
        plt.legend()
        plt.xlabel("dV")
        plt.ylabel("dS")
        plt.tight_layout()
        plt.show()
        dfs_count += 1


main()
