import pandas as pd
import numpy as np
import re
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
    find_thrust(dfs)
    # subumbrellar_to_bell(dfs, oris)

    for df in dfs:
        print(df)


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


######################################################################
# find the modeled thrust force based on the modeled acceleration
# derived by the basic measurements
######################################################################
def find_thrust(dfs_ref):
    for df in dfs_ref:
        volumes = []  # store instantaneous volumes
        masses = []  # store instantaneous masses
        orifices = []  # store instantaneous orifices
        drags = []  # store instantaneous drags
        # net_forces = []  # store instantaneous net_forces
        # thrusts = []  # store instantaneous thrusts

        for row in df.index:
            h = df.at[row, 'h']
            d = df.at[row, 'd']
            r = df.at[row, 're']
            u = df.at[row, 'u']
            volumes.append(vol(h, d))
            masses.append(mas(h, d))
            drags.append(drg(r, h, d, u))
            orifices.append(ori(d))
            # net_forces.append(nfr_m(h, d, am))
            # thrusts.append(thr_m(h, d, am, r, u))

        df["vol"] = volumes
        df["mass"] = masses
        df["drag"] = drags
        df["ori"] = orifices
        # df["modeled_net"] = net_forces

        # df["modeled thrust"] = thrusts


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
        for row in (list(range(len(df.index) - 1))):
            f = df.at[row, thrust]
            v1 = df.at[row, 'vol']
            v2 = df.at[row + 1, 'vol']
            t = df.at[0, 'st']
            dVdt.append((v2 - v1) / t)
            dSdt.append(np.sqrt(ori_ref[count] / sea_den * f))
            SV = dsdv(f, ori_ref[count], v1, v2, t)
            dSdV.append(SV)
            dV.append(v2 - v1)
            dS.append(SV * (v2 - v1))

        dVdt.append(0)
        dSdt.append(0)
        dSdV.append(0)
        dV.append(0)
        dS.append(0)

        # df["dSdt"] = dSdt
        # df["dVdt"] = dVdt
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
    dvdt = (vol_2 - vol_1) / t
    dsdv = np.absolute(dsdt / dvdt)
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


######################################################################
# effective mass
######################################################################
def s():
    return 0


######################################################################
# corrected fluid ejection thrust
######################################################################
def thr_p(d_ref, surf_1, surf_2, t_1, t_2):
    area = ori(d_ref)
    dsdt = (surf_2 - surf_1) / (t_2 - t_1)
    return sea_den / area * np.power(dsdt, 2)


main()
