import pandas as pd
import numpy as np
import re

# global variables

f_path = "/Users/beelee/PycharmProjects/OblateThrust/csv/"
a_digitale_f = f_path + "a_digitale_data.csv"
s_sp_f = f_path + "s_sp_data.csv"
p_flavicirrata_f = f_path + "p_flavicirrata_data.csv"
a_victoria_f = f_path + "a_victoria_data.csv"
m_cellularia_f = f_path + "m_cellularia_data.csv"
p_gregarium_f = f_path + "p_gregarium_data.csv"


vel_o = 0
acc_m = 0
acc_o = 0

s_b = 0.0

t_rel = 0
t_pau = 0
t_con = 0
t_dri = 0


sea_den = 1.024 * np.power(10.0, 6)  # g/m^3, 1.024 g/cm^3 (Colin & Costello, 2001)
sea_vis = np.power(10.0, -6)  # m^2/s

F_s = 0

acc_cor = 0


def main():

    # import pre-cleaned up csv files, 1 prolate 2 oblates
    a_digitale = pd.read_csv(a_digitale_f)
    s_sp = pd.read_csv(s_sp_f)
    p_flavicirrata = pd.read_csv(p_flavicirrata_f)
    a_victoria = pd.read_csv(a_victoria_f)
    m_cellularia = pd.read_csv(m_cellularia_f)
    p_gregarium = pd.read_csv(p_gregarium_f)
    dfs = [a_digitale, s_sp, p_flavicirrata, a_victoria, m_cellularia, p_gregarium]

    # clean up dataframes

    a_digitale.dropna(axis=0, how='any', inplace=True)  # drop incomplete rows for the prolate medusae
    m_cellularia.dropna(axis=0, how='any', inplace=True)  # drop incomplete rows for the oblate medusae

    # for column in dfs_2001[0].columns:  # shift velocity columns to line up data for prolate medusae
    #     index_no = dfs_2001[0].columns.get_loc(column)
    #     if index_no == 5 or index_no == 6:
    #         dfs_2001[0][column] = dfs_2001[0][column].shift(-1)

    # deleate extra references
    for df in dfs:

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

    for df in dfs:
        f_col = ''
        vel_col = ''
        re_col = ''

        for column in df.columns:
            if re.search(r'f', column):
                f_col = re.search(r'f', column).string
                # print(f_col)
            if re.search(r'v', column):
                vel_col = re.search(r'v', column).string
                # print(vel_col)
            if re.search(r're', column):
                re_col = re.search(r're', column).string
                # print(re_col)

        velocities = []  # store instantaneous velocities with newly converted units
        heights = []  # store instantaneous heights
        diameters = []  # store instantaneous diameters

        for row in df.index:
            u = df.at[row, vel_col] / 100.0  # convert velocity unit to m/s
            velocities.append(u)
            d_h = dim(df.at[row, re_col], u, df.at[row, f_col])  # re: m^2/s / m^2/s, fineness: m/m
            diameters.append(d_h[0])
            heights.append(d_h[1])

        df["u"] = velocities
        df["h"] = heights
        df["d"] = diameters


    for df in dfs:
        volumes = []  # store instantaneous volumes
        masses = []  # store instantaneous masses
        orifices = []  # store instantaneous orifices
        drags = []  # store instantaneous drags
        net_forces = []  # store instantaneous net_forces
        thrusts = []  # store instantaneous thrusts

        for row in df.index:
            h = df.at[row, 'h']
            d = df.at[row, 'd']
            r = df.at[row, 're']
            u = df.at[row, 'u']
            volumes.append(vol(h, d))
            masses.append(mas(h, d))
            orifices.append(ori(d))
            drags.append(drg(h, d, r, u))
            net_forces.append(nfr_m(h, d))
            thrusts.append(thr_m(h, d, r, u))

        df["vol"] = volumes
        df["mass"] = masses
        df["ori"] = orifices
        df["drag"] = drags
        df["modeled_net"] = net_forces
        df["modeled thrust"] = thrusts

        print(df)


######################################################################
# set bell diameter (m) and height (m)
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
    coe = d_ref / (2 * np.power(h_ref, 1.4))
    # effective mass = sea density * bell volume * (1 + mass coefficient)
    # mass: g = g/m^3 * m^3
    mass = sea_den * volume * (1 + coe)
    return mass


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
# get bell fineness
# param: bell height(m), bell diameter(m)
######################################################################
def fin(h_ref, d_ref):
    # bell fineness = bell height / bell diameter
    # fineness: m/m
    fineness = h_ref / d_ref
    return fineness


######################################################################
# get net force according to observation (g * m / s^2)
######################################################################
def nfr_m(h_ref, d_ref):
    mass = mas(h_ref, d_ref)
    # net_force =
    return acc_m * mass


######################################################################
# modeled thrust
######################################################################
def thr_m(h_ref, d_ref, re_ref, u_ref):
    force = nfr_m(h_ref, d_ref)
    drag = drg(h_ref, d_ref, re_ref, u_ref)
    return force + drag


######################################################################
# change of subumbrellar volume with respect to the change of
# bell volume
######################################################################
def dsdv(thrust_ref, d_ref, vol_1, vol_2, t_1, t_2):
    area = ori(d_ref)
    dsdt = np.sqrt(area / sea_den * thrust_ref)
    dvdt = (vol_2 - vol_1)/(t_2 - t_1)
    return dsdt / dvdt


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
    dsdt = (surf_2 - surf_1)/(t_2 - t_1)
    return sea_den / area * np.power(dsdt, 2)


main()
