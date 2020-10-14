import pandas as pd
import numpy as np
import re
from matplotlib import pyplot as plt

# global variables

f_path = "/Users/beelee/PycharmProjects/OblateThrust/csv/"
s_sp_f = f_path + "s_sp_data.csv"
p_gregarium_f = f_path + "p_gregarium_data.csv"

sea_den = 1.024 * np.power(10.0, 6)  # g/m^3, 1.024 g/cm^3 (Colin & Costello, 2001)
sea_vis = np.power(10.0, -6)  # m^2/s


######################################################################
# task 1: subumbrellar volume estimator
######################################################################
def main():

    # import pre-cleaned up csv files, 1 prolate 1 oblates
    s_sp = pd.read_csv(s_sp_f)
    p_gregarium = pd.read_csv(p_gregarium_f)

    # group the two medusae of interest for later use
    dfs = [s_sp, p_gregarium]


    # group constant orifice areas for later use
    # 0.85cm & 2.14cm bell diameters for s_sp and p_gregarium (Colin & Costello, 2001)
    s_sp_ori = ori(0.85 / 100)
    p_gregarium_ori = ori(2.14 / 100)
    oris = [s_sp_ori, p_gregarium_ori]

    # clean up dataframes
    # a_digitale.dropna(axis=0, how='any', inplace=True)  # drop incomplete rows for the prolate medusae
    # m_cellularia.dropna(axis=0, how='any', inplace=True)  # drop incomplete rows for the oblate medusae

    # for column in dfs_2001[0].columns:  # shift velocity columns to line up data for prolate medusae
    #     index_no = dfs_2001[0].columns.get_loc(column)
    #     if index_no == 5 or index_no == 6:
    #         dfs_2001[0][column] = dfs_2001[0][column].shift(-1)

    clean_time(dfs)
    add_basics(dfs)
    mod_thrust(dfs)
    subumbrellar_to_bell(dfs, oris)

    # split and align pulsations data for each medusae
    s_sp_1 = s_sp[1:13].copy()
    s_sp_2 = s_sp[12:25].copy()
    s_sp_2["st"] = s_sp_2["st"] - (s_sp["st"][12] - (s_sp["st"][0]+s_sp["st"][1])/2)
    s_sp_3 = s_sp[24:36].copy()
    s_sp_3["st"] = s_sp_3["st"] - (s_sp["st"][24] - s_sp["st"][1])
    s_sp_4 = s_sp[35:45].copy()
    s_sp_4["st"] = s_sp_4["st"] - (s_sp["st"][35] - s_sp["st"][2])
    s_sps = [s_sp_1, s_sp_2, s_sp_3, s_sp_4]

    p_gregarium_1 = p_gregarium[1:6].copy()
    p_gregarium_2 = p_gregarium[5:9].copy()
    p_gregarium_2["st"] = p_gregarium_2["st"] - (p_gregarium["st"][5] - (p_gregarium["st"][1] + p_gregarium["st"][2])/2)
    p_gregarium_3 = p_gregarium[8:13].copy()
    p_gregarium_3["st"] = p_gregarium_3["st"] - (p_gregarium["st"][8] - p_gregarium["st"][1])
    p_gregarium_4 = p_gregarium[12:17].copy()
    p_gregarium_4["st"] = p_gregarium_4["st"] - (p_gregarium["st"][12] - p_gregarium["st"][1])
    p_gregariums = [p_gregarium_1, p_gregarium_2, p_gregarium_3, p_gregarium_4]
    dfss = [s_sps, p_gregariums]

    medusae = ['prolate', 'oblate']
    dfs_count = 0
    for dfs in dfss:
        total_x = np.array([])
        total_y = np.array([])
        colors = ['goldenrod', 'firebrick', 'forestgreen', 'dodgerblue']
        df_count = 0
        for df in dfs:
            label = "cycle #" + str(df_count+1)
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

    dfs_count = 0
    for dfs in dfss:
        total_x = np.array([])
        total_y = np.array([])
        for df in dfs:
            total_x = np.append(total_x, df["dV"].values)
            total_y = np.append(total_y, df["dS"].values)
        polymodel2 = np.poly1d(np.polyfit(total_x, total_y, 2))
        polyline2 = np.linspace(min(total_x), max(total_x), 100)
        plt.scatter(total_x, total_y)
        plt.plot(polyline2, polymodel2(polyline2), label='dV to dS regression')
        plt.title("change of %s subumbrellar volume with respect to change of bell volume" % (medusae[dfs_count]))
        plt.legend()
        plt.xlabel("dV")
        plt.ylabel("dS")
        plt.tight_layout()
        plt.show()

        dfs_count += 1


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
        # f_col = ''
        # vel_col = ''
        # re_col = ''
        #
        # for column in df.columns:
        #     if re.search(r'f', column):
        #         f_col = re.search(r'f', column).string
        #         # print(f_col)
        #     if re.search(r'v', column):
        #         vel_col = re.search(r'v', column).string
        #         # print(vel_col)
        #     if re.search(r're', column):
        #         re_col = re.search(r're', column).string
        #         # print(re_col)
        velocities = []  # store instantaneous velocities in converted units
        accelerations = []  # store instantaneous accelerations in converted units
        heights = []  # store instantaneous heights
        diameters = []  # store instantaneous diameters

        for row in df.index:
            u = df.at[row, 'v'] / 100.0  # convert velocity unit to m/s
            velocities.append(u)
            amc = df.at[row, 'am'] / 100.0
            accelerations.append(amc)
            d_h = dim(df.at[row, 're'], u, df.at[row, 'f'])  # re: m^2/s / m^2/s, fineness: m/m
            diameters.append(d_h[0])
            heights.append(d_h[1])

        df["u"] = velocities
        df["amc"] = accelerations
        df["h"] = heights
        df["d"] = diameters


######################################################################
# find the modeled thrust force based on the modeled acceleration
# derived by the basic measurements
######################################################################
def mod_thrust(dfs_ref):
    for df in dfs_ref:
        volumes = []  # store instantaneous volumes
        masses = []  # store instantaneous masses
        # orifices = []  # store instantaneous orifices
        drags = []  # store instantaneous drags
        net_forces = []  # store instantaneous net_forces
        thrusts = []  # store instantaneous thrusts

        for row in df.index:
            h = df.at[row, 'h']
            d = df.at[row, 'd']
            am = df.at[row, 'amc']
            r = df.at[row, 're']
            u = df.at[row, 'u']
            volumes.append(vol(h, d))
            masses.append(mas(h, d))
            net_forces.append(nfr_m(h, d, am))
            drags.append(drg(r, h, d, u))
            thrusts.append(thr_m(h, d, am, r, u))

        df["vol"] = volumes
        df["mass"] = masses
        df["modeled_net"] = net_forces
        df["drag"] = drags
        df["modeled thrust"] = thrusts


######################################################################
# find the modeled thrust force based on the modeled acceleration
# derived by the basic measurements
######################################################################
def subumbrellar_to_bell(dfs_ref, ori_ref):
    count = 0
    for df in dfs_ref:
        dSdt = []
        dVdt = []
        dSdV = []  # store instantaneous volumes
        dV = []
        dS = []

        for column in df.columns:
            if re.search(r'thrust', column):
                thrust = re.search(r'thrust', column).string
        for row in (list(range(len(df.index)-1))):
            f = df.at[row, thrust]
            v1 = df.at[row, 'vol']
            v2 = df.at[row+1, 'vol']
            t = df.at[0, 'st']
            dVdt.append((v2 - v1) / t)
            if (v2 - v1) / t < 0:
                dSdt.append(-1 * np.sqrt(ori_ref[count] / sea_den * f))
            else:
                dSdt.append(np.sqrt(ori_ref[count] / sea_den * f))
            SV = dsdv(f, ori_ref[count], v1, v2, t)
            dSdV.append(SV)
            dV.append(v2-v1)
            dS.append(SV * (v2-v1))

        dVdt.append(0)
        dSdt.append(0)
        dSdV.append(0)
        dV.append(0)
        dS.append(0)

        df["dVdt"] = dVdt
        df["dSdt"] = dSdt
        df["dSdV"] = dSdV
        df["dV"] = dV
        df["dS"] = dS

        count += 1


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
    volume = 2/3 * h_ref * np.pi * np.power(radius, 2)
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
# get net force (g * m / s^2)
# param: bell height(m), bell diameter(m), modeled acceleration(m/s^2)
######################################################################
def nfr_m(h_ref, d_ref, am_ref):
    mass = mas(h_ref, d_ref)
    # force = mass * acceleration
    # force: g * m / s^2 = g * (m / s^2)
    net_force = mass * am_ref
    return net_force


######################################################################
# get drag (g * m / s^2)
# param: Re, bell height(m), bell diameter(m), swimming velocity (m/s)
######################################################################
def drg(re_ref, h_ref, d_ref, u_ref):
    coe = 0
    if re_ref > 500:
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

    return drag


######################################################################
# get thrust (g * m / s^2)
# param: bell height(m), bell diameter(m), acceleration(m/s^2),
#        Re, swimming velocity (m/s)
######################################################################
def thr_m(h_ref, d_ref, am_ref, re_ref, u_ref):
    force = nfr_m(h_ref, d_ref, am_ref)
    drag = drg(re_ref, h_ref, d_ref, u_ref)
    thrust = force + drag
    if thrust < 0:
        return 0
    else:
        return thrust


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
    dvdt = (vol_2 - vol_1)/t
    dsdv = np.absolute(dsdt/dvdt)
    return dsdv


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