import pandas as pd
import numpy as np
import re
from matplotlib import pyplot as plt
from matplotlib.pyplot import figtext

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


######################################################################
# split s_sp dataframe into 4 based on volume
# param: df
######################################################################
def split_sp(df_ref):
    ref_1 = df_ref[3:13].copy()
    ref_1['st'] = np.arange(0, df_ref.at[9, 'st']+0.05, df_ref.at[9, 'st'] / 9)
    ref_2 = df_ref[13:26].copy()
    ref_2['st'] = np.arange(0, df_ref.at[9, 'st']+0.05, df_ref.at[9, 'st'] / 12)
    ref_3 = df_ref[26:36].copy()
    ref_3['st'] = np.arange(0, df_ref.at[9, 'st']+0.05, df_ref.at[9, 'st'] / 9)
    ref_4 = df_ref[36:45].copy()
    ref_4['st'] = np.arange(0, df_ref.at[9, 'st']+0.05, df_ref.at[9, 'st'] / 8)

    return [ref_1, ref_2, ref_3, ref_4]


######################################################################
# split p_gregarium dataframe into 4 based on volume
# param: df
######################################################################
def split_gregarium(df_ref):
    ref_1 = df_ref[2:5].copy()
    ref_1['st'] = np.arange(0, df_ref.at[2, 'st'] + 0.05, df_ref.at[2, 'st'] / 2)
    ref_2 = df_ref[5:9].copy()
    ref_2['st'] = np.arange(0, df_ref.at[2, 'st'] + 0.05, df_ref.at[2, 'st'] / 3)
    ref_3 = df_ref[9:13].copy()
    ref_3['st'] = np.arange(0, df_ref.at[2, 'st'] + 0.05, df_ref.at[2, 'st'] / 3)
    ref_4 = df_ref[13:17].copy()
    ref_4['st'] = np.arange(0, df_ref.at[2, 'st'] + 0.05, df_ref.at[2, 'st'] / 3)
    return [ref_1, ref_2, ref_3, ref_4]

######################################################################
# clean up dataframe to show just 1 equivalent time reference
######################################################################
def clean_time(df_ref):
    # store columns to delete
    to_delete = []
    for column in df_ref.columns:
        # get the column index number
        index_no = df_ref.columns.get_loc(column)

        # besides standard time in column 0, all time data are in odd columns
        # --> inspect every odd columns
        if index_no % 2 == 1:
            # this column is a time reference
            eq = True
            # assume no more than 2 references in a row are considerably different to standard time
            second_err = False
            for row in df_ref.index:
                if np.absolute(df_ref.iat[row, 0] - round(df_ref.iat[row, index_no], 2)) > 0.02:
                    if second_err:
                        break
                    second_err = True
                else:
                    second_err = False
        else:
            eq = False

        if eq:
            to_delete.append(column)

    #  delete the deletable columns
    if to_delete:
        df_ref.drop(to_delete, axis=1, inplace=True)


######################################################################
# add/adjust instantaneous heights, diameters, and velocities and
# accelerations in the corrected units
######################################################################
def get_basics(df_ref):
    accelerations_o = []
    accelerations_m = []
    velocities = []
    heights = []
    diameters = []

    for row in df_ref.index:
        aoc = df_ref.at[row, 'ao'] / 100.0
        accelerations_o.append(aoc)
        amc = df_ref.at[row, 'am'] / 100.0
        accelerations_m.append(amc)
        u = df_ref.at[row, 'v'] / 100.0  # convert velocity unit to m/s
        velocities.append(u)
        d_h = bell_dim(df_ref.at[row, 're'], u, df_ref.at[row, 'f'])  # re: m^2/s / m^2/s, fineness: m/m
        diameters.append(d_h[0])
        heights.append(d_h[1])

    df_ref["ao"] = accelerations_o
    df_ref["am"] = accelerations_m
    df_ref["v"] = velocities
    df_ref["h"] = heights
    df_ref["d"] = diameters


######################################################################
# find the modeled thrust force based on the modeled acceleration
# derived by the basic measurements
######################################################################
def get_thrust(df_ref):
    volumes = []
    masses = []
    drags = []
    net_forces = []
    thrusts = []

    for row in df_ref.index:
        am = df_ref.at[row, 'am']
        v = df_ref.at[row, 'v']
        r = df_ref.at[row, 're']
        h = df_ref.at[row, 'h']
        d = df_ref.at[row, 'd']
        volumes.append(bell_vol(h, d))
        masses.append(bell_mas(h, d))
        net_forces.append(nf_am(h, d, am))
        drags.append(drg(r, h, d, v))
        thrusts.append(tf_am(h, d, am, r, v))

    df_ref["V"] = volumes
    df_ref["tf"] = thrusts


######################################################################
# find the modeled thrust force based on the modeled acceleration
# derived by the basic measurements
######################################################################
def sub_n_vol_change(df_ref, ori_ref):
    thrust = ""
    dV = []  # store instantaneous dV
    dS = []  # store instantaneous dS
    # dSdt = []  # store instantaneous dSdt
    # dVdt = []  # store instantaneous dVdt
    # dSdV = []  # store instantaneous dSdV

    for column in df_ref.columns:
        if re.search(r'tf', column):
            thrust = re.search(r'tf', column).string
    for row in (list(range(len(df_ref.index) - 1))):
        t = df_ref.at[0, 'st']
        v1 = df_ref.at[row, 'V']
        v2 = df_ref.at[row + 1, 'V']
        dV.append(v2 - v1)
        # dVdt.append((v2 - v1) / t)
        f = df_ref.at[row, thrust]
        # if v1 > v2:
        #     dSdt.append(-1 * np.sqrt(ori_ref / sea_den * f))
        # else:
        #     dSdt.append(np.sqrt(ori_ref / sea_den * f))
        SV = dsdv(f, ori_ref, v1, v2, t)
        # dSdV.append(SV)
        dS.append(SV * (v2 - v1))

    dV.append(0)
    dS.append(0)
    # dVdt.append(0)
    # dSdt.append(0)
    # dSdV.append(0)

    df_ref["dV"] = dV
    df_ref["dS"] = dS
    # df_ref["dVdt"] = dVdt
    # df_ref["dSdt"] = dSdt
    # df_ref["dSdV"] = dSdV


######################################################################
# get bell diameter (m) and height (m)
# param: Re, velocity(m/s), fineness
######################################################################
def bell_dim(re_ref, u_ref, f_ref):
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
def bell_vol(h_ref, d_ref):
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
def bell_mas(h_ref, d_ref):
    volume = bell_vol(h_ref, d_ref)
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
def nf_am(h_ref, d_ref, am_ref):
    mass = bell_mas(h_ref, d_ref)
    # force = mass * acceleration
    # force: g * m / s^2 = g * (m / s^2)
    net_force = mass * am_ref
    return net_force


######################################################################
# get drag (g * m / s^2)
# param: Re, bell height(m), bell diameter(m), swimming velocity (m/s)
######################################################################
def drg(re_ref, h_ref, d_ref, u_ref):
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
def tf_am(h_ref, d_ref, am_ref, re_ref, u_ref):
    force = nf_am(h_ref, d_ref, am_ref)
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
