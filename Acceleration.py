import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# global variables

f_path = "/Users/beelee/PycharmProjects/OblateThrust/csv/"
# a_digitale_f = f_path + "a_digitale_data.csv"
# s_sp_f = f_path + "s_sp_data.csv"
# p_flavicirrata_f = f_path + "p_flavicirrata_data.csv"
a_victoria_f = f_path + "a_victoria_data.csv"
m_cellularia_f = f_path + "m_cellularia_data.csv"
p_gregarium_f = f_path + "p_gregarium_data.csv"

sea_den = 1.024 * np.power(10.0, 6)  # g/m^3, 1.024 g/cm^3 (Colin & Costello, 2001)
sea_vis = np.power(10.0, -6)  # m^2/s


######################################################################
# task 2: model accurate oblate medusae acceleration
######################################################################
def main():

    # import pre-cleaned up csv files, 3 oblates
    a_victoria = pd.read_csv(a_victoria_f)
    m_cellularia = pd.read_csv(m_cellularia_f)
    p_gregarium = pd.read_csv(p_gregarium_f)

    # group the two medusae of interest for later use
    dfs = [a_victoria, m_cellularia, p_gregarium]

    clean_time(dfs)
    p_gregarium.drop('am', axis='columns', inplace=True)
    add_basics(dfs)
    find_accel(dfs)
    for df in dfs:
        df.drop(df.tail(1).index, inplace=True)

    # split and align pulsations data for each medusae
    a_victoria_1 = a_victoria[1:11].copy()
    a_victoria_2 = a_victoria[10:19].copy()
    a_victoria_2["st"] = a_victoria_2["st"] - (a_victoria["st"][10] - a_victoria["st"][3])
    a_victoria_3 = a_victoria[18:26].copy()
    a_victoria_3["st"] = a_victoria_3["st"] - (a_victoria["st"][18] - a_victoria["st"][3])
    a_victoria_4 = a_victoria[25:32].copy()
    a_victoria_4["st"] = a_victoria_4["st"] - (a_victoria["st"][25] - (a_victoria["st"][3] + a_victoria["st"][4])/2)
    a_victorias = [a_victoria_1, a_victoria_2, a_victoria_3, a_victoria_4]

    m_cellularia_1 = m_cellularia[4:17].copy()
    m_cellularia_1["st"] = m_cellularia_1["st"] + m_cellularia["st"][0]/2
    m_cellularia_2 = m_cellularia[16:29].copy()
    m_cellularia_2["st"] = m_cellularia_2["st"] - (m_cellularia["st"][16] - m_cellularia["st"][5])
    m_cellularia_3 = m_cellularia[28:40].copy()
    m_cellularia_3["st"] = m_cellularia_3["st"] - (m_cellularia["st"][28] - m_cellularia["st"][6])
    m_cellularia_4 = m_cellularia[39:55].copy()
    m_cellularia_4["st"] = m_cellularia_4["st"] - (m_cellularia["st"][39] - m_cellularia["st"][4])
    m_cellularias = [m_cellularia_1, m_cellularia_2, m_cellularia_3, m_cellularia_4]

    p_gregarium_1 = p_gregarium[1:6].copy()
    p_gregarium_1["st"] = p_gregarium_1["st"] + (p_gregarium["st"][0] / 2)
    p_gregarium_2 = p_gregarium[5:9].copy()
    p_gregarium_2["st"] = p_gregarium_2["st"] - (p_gregarium["st"][5] - (p_gregarium["st"][1] + p_gregarium["st"][2]) / 2)
    p_gregarium_3 = p_gregarium[8:13].copy()
    p_gregarium_3["st"] = p_gregarium_3["st"] - (p_gregarium["st"][8] - p_gregarium["st"][1])
    p_gregarium_4 = p_gregarium[12:17].copy()
    p_gregarium_4["st"] = p_gregarium_4["st"] - (p_gregarium["st"][12] - p_gregarium["st"][1])
    p_gregariums = [p_gregarium_1, p_gregarium_2, p_gregarium_3, p_gregarium_4]
    dfss = [a_victorias, m_cellularias, p_gregariums]

    medusae = ['a_victoria', 'm_cellularia', 'p_gregarium']
    dfs_count = 0
    for dfs in dfss:
        total_x = np.array([])
        total_y = np.array([])
        colors = ['goldenrod', 'firebrick', 'forestgreen', 'dodgerblue']
        df_count = 0
        for df in dfs:
            label = "cycle #" + str(df_count + 1)
            plt.plot(df["st"], df["vol"], color=colors[df_count], label=label)
            total_x = np.append(total_x, df["st"].values)
            total_y = np.append(total_y, df["vol"].values)
            df_count += 1
        polymodel1 = np.poly1d(np.polyfit(total_x, total_y, 3))
        polyline1 = np.linspace(min(total_x), max(total_x), 100)
        plt.plot(polyline1, polymodel1(polyline1), color='magenta', linestyle='--', label='p(3) regression')
        plt.title("%s bell volume over 1 pulsation cycle" % (medusae[dfs_count]))
        plt.legend()
        plt.xlabel("time s")
        plt.ylabel("volume m^3")
        plt.tight_layout()
        plt.show()

        dfs_count += 1

    for dfs in dfss:
        for df in dfs:
            for row in df.index:
                print("row %i value %f" % (row, df.loc[row, 'am']))
                if df.loc[row, 'am'] > 100:
                    print("row %i value %f" % (row, df.loc[row, 'am']))
                    df.drop(row, inplace=True)

    dfs_count = 0
    for dfs in dfss:
        total_x = np.array([])
        total_y = np.array([])
        colors = ['goldenrod', 'firebrick', 'forestgreen', 'dodgerblue']
        df_count = 0
        for df in dfs:
            label = "cycle #" + str(df_count + 1)
            plt.plot(df["st"], df["am"], color=colors[df_count], label=label)
            total_x = np.append(total_x, df["st"].values)
            total_y = np.append(total_y, df["am"].values)
            df_count += 1
        polymodel2 = np.poly1d(np.polyfit(total_x, total_y, 2))
        polyline2 = np.linspace(min(total_x), max(total_x), 100)
        plt.plot(polyline2, polymodel2(polyline2), color='magenta', linestyle='--', label='modeled p(2) regression')
        plt.title("%s acceleration over time" % (medusae[dfs_count]))
        plt.legend()
        plt.xlabel("time s")
        plt.ylabel("acceleration ms^-2")
        plt.tight_layout()
        plt.show()

        # print(polymodel2)

    # for df in dfs:
    #     print(df)


######################################################################
# clean up data frame to show only 1 corrected time reference
######################################################################
def clean_time(dfs_ref):
    for df in dfs_ref:
        to_delete = []  # store columns to delete as a list
        for column in df.columns:
            index_no = df.columns.get_loc(column)  # get the column index number
            # print("column #%i" % index_no)

            if index_no % 2 == 1:
                eq = True
                second_err = False  # only cancel deletion when 2 time references in a row are different from template
                # print("eq set to true")
                for row in df.index:
                    if np.absolute(df.iat[row, 0] - round(df.iat[row, index_no], 2)) > 0.02:
                        # print("row value %f and column value %f" % (df.iat[row, 0], round(df.iat[row, index_no], 2)))
                        if second_err:
                            eq = False
                            # print("eq turns false")
                            break
                        second_err = True
                    else:
                        second_err = False
            else:
                eq = False
                # print("eq set to false")

            if eq:
                # print("eq true all the way at column %i, so store name" % index_no)
                to_delete.append(column)

        if to_delete:
            df.drop(to_delete, axis=1, inplace=True)


######################################################################
# add instantaneous heights, diameters, and velocities and accelerations
# in the corrected units. these complete the basic measurements
######################################################################
def add_basics(dfs_ref):
    for df in dfs_ref:

        velocities = []  # store instantaneous velocities in converted units
        heights = []  # store instantaneous heights
        diameters = []  # store instantaneous diameters

        for row in df.index:
            u = df.at[row, 'v'] / 100.0  # convert velocity unit to m/s
            velocities.append(u)
            d_h = dim(df.at[row, 're'], u, df.at[row, 'f'])  # re: m^2/s / m^2/s, fineness: m/m
            diameters.append(d_h[0])
            heights.append(d_h[1])

        df["u"] = velocities
        df["h"] = heights
        df["d"] = diameters


######################################################################
# estimate change of oblate subumbrellar volume
######################################################################
def dS(dV):
    # formula acquired from Sub_volume.py
    res = 8064 * np.power(dV, 2) + 0.1188 * dV + 1.217e-07
    return res


######################################################################
# find the modeled thrust force based on the modeled acceleration
# derived by the basic measurements
######################################################################
def find_accel(dfs_ref):
    for df in dfs_ref:
        volumes = []  # store instantaneous volumes
        masses = []  # store instantaneous masses
        orifices = []  # store instantaneous orifices
        drags = []  # store instantaneous drags
        dSdt = []
        thrusts = []  # store instantaneous thrusts
        net_forces = []  # store instantaneous net_forces
        accelerations = []

        for row in df.index:
            h = df.at[row, 'h']
            d = df.at[row, 'd']
            r = df.at[row, 're']
            u = df.at[row, 'u']
            volumes.append(vol(h, d))
            masses.append(mas(h, d))
            drags.append(drg(r, h, d, u))
            orifices.append(ori(d))

        df["vol"] = volumes
        df["ori"] = orifices

        for row in list(range(len(df.index)-1)):
            v1 = df.at[row, 'vol']
            v2 = df.at[row+1, 'vol']
            t1 = df.at[row, 'st']
            t2 = df.at[row + 1, 'st']
            o = df.at[row, 'ori']
            dSdt.append(dS(v2-v1)/(t2-t1))
            thrusts.append(thr_p(o, v1, v2, t1, t2))

        dSdt.append(0)
        thrusts.append(0)

        df["drag"] = drags
        df["dSdt"] = dSdt
        df["thrust"] = thrusts

        for row in df.index:
            thr = df.at[row, 'thrust']
            drag = df.at[row, 'drag']
            net_forces.append(nfr(thr, drag))

        df["mass"] = masses
        df["force"] = net_forces

        for row in df.index:
            m = df.at[row, 'mass']
            f = df.at[row, 'force']
            accelerations.append(f/m)

        df["am"] = accelerations


######################################################################
# get bell diameter (m) and height (m)
# param: Re, velocity(m/s), fineness
######################################################################
def dim(re_ref, u_ref, f_ref):
    # bell diameter = Re * sea kinematic viscosity / swimming velocity
    # diameter: m = (m^2/s / m^2/s) (m^2/s) / m/s
    d_b = 1.0 * re_ref * sea_vis / np.absolute(u_ref)
    # bell height = fineness * bell diameter
    # height: m = (m/m) * m
    h_b = d_b * f_ref
    return [d_b, h_b]


######################################################################
# get bell volume (m^3)
# param: bell height(m), bell diameter(m)
######################################################################
def vol(h_ref, d_ref):
    # bell radius = bell diameter / 2
    # radius: m
    radius = d_ref / 2
    # bell volume = 2/3 * bell height * pi * radius^2
    # volume: m^3 = m * m^2
    volume = 2 / 3 * h_ref * np.pi * np.power(radius, 2)
    return volume


######################################################################
# get effective mass (g)
# param: bell height(m), bell diameter(m)
######################################################################
def mas(h_ref, d_ref):
    volume = vol(h_ref, d_ref)
    # mass coefficient = bell diameter / 2 * bell height^(1.4)
    # coefficient = m/m
    coe = np.power(d_ref / 2 / h_ref, 1.4)
    # effective mass = sea density * bell volume * (1 + mass coefficient)
    # mass: g = g/m^3 * m^3
    mass = sea_den * volume * (1 + coe)
    return mass


######################################################################
# get drag (g * m / s^2)
# param: Re, bell height(m), bell diameter(m), swimming velocity (m/s)
######################################################################
def drg(re_ref, h_ref, d_ref, u_ref):
    if re_ref > 700:
        return 0
    elif re_ref < 1:
        # drag coefficient = 24 / re
        # coefficient:
        coe = 24 / re_ref
    else:
        # drag coefficient = 24 / re^0.7
        # coefficient:
        coe = 24 / np.power(re_ref, 0.7)

    # bell surface area = pi * bell height * bell diameter / 4
    # area: m^2 = m * m
    area = np.pi * h_ref * d_ref / 4

    # drag force = sea density * swimming velocity^2 * bell surface area / 2
    # drag: g * m / s^2 = g/m^3 * (m/s)^2 * m^2
    drag = (sea_den * np.power(u_ref, 2) * area * coe) / 2

    if drag < 0.000001:
        print("coe %f area %f veloc %f" % (coe, area, u_ref))

    return drag


######################################################################
# get orifice area (m^2)
# param: bell diameter(m)
######################################################################
def ori(d_ref):
    # bell radius = bell diameter / 2
    # radius: m
    radius = d_ref / 2
    # orifice area = pi * bell radius^2
    # area: m^2
    area = np.power(radius, 2) * np.pi
    return area


######################################################################
# change of subumbrellar volume with respect to the change of
# bell volume
######################################################################
def dsdv(thrust_ref, ori_ref, vol_1, vol_2, t):
    # this calculated dsdt value will always be positive, but it is
    # negative when dv when dvdt is negative, which signifies a contracting
    # bell. therefore, dsdv should always be positive
    dsdt = np.sqrt(ori_ref / sea_den * thrust_ref)
    dvdt = (vol_2 - vol_1) / t
    res = np.absolute(dsdt / dvdt)
    return res


######################################################################
# corrected fluid ejection thrust
######################################################################
def thr_p(ori_ref, v1_ref, v2_ref, t_1, t_2):
    dsdt = dS(v2_ref - v1_ref) / (t_2 - t_1)
    thrust = sea_den / ori_ref * np.power(dsdt, 2)
    return thrust


######################################################################
# get net force (g * m / s^2)
# param: thrust (g * m / s^2), drag (g * m / s^2)
######################################################################
def nfr(thr_ref, drg_ref):
    # force = thrust - drag
    # force: g * m / s^2 = g * m / s^2 - g * m / s^2
    net_force = thr_ref - drg_ref
    return net_force


######################################################################
# get bell fineness
# param: bell height(m), bell diameter(m)
######################################################################
def fin(h_ref, d_ref):
    # bell fineness = bell height / bell diameter
    # fineness: m/m
    fineness = h_ref / d_ref
    return fineness


main()
