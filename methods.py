import pandas as pd
import numpy as np
import re
from matplotlib import pyplot as plt
from matplotlib.pyplot import figtext
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import researchpy as rp
import statsmodels.api as sm
import scipy.stats as stats

f_path = "/Users/beelee/PycharmProjects/OblateThrust/csv/"
a_digitale_f = f_path + "a_digitale_data.csv"
s_sp_f = f_path + "s_sp_data.csv"
p_flavicirrata_f = f_path + "p_flavicirrata_data.csv"
a_victoria_f = f_path + "a_victoria_data.csv"
m_cellularia_f = f_path + "m_cellularia_data.csv"
p_gregarium_f = f_path + "p_gregarium_data.csv"
a_aurita_f = f_path + "a_aurita_data.csv"
e_indicans_f = f_path + "e_indicans_data.csv"
s_meleagris_f = f_path + "s_meleagris_data.csv"
l_unguiculata_f = f_path + "l_unguiculata_data.csv"
c_capillata_f = f_path + "c_capillata_data.csv"
c_capillata_2_f = f_path + "c_capillata_2_data.csv"
l_tetraphylla_f = f_path + "l_tetraphylla_data.csv"
l_tetraphylla_2_f = f_path + "l_tetraphylla_2_data.csv"
all_csv = [a_digitale_f, s_sp_f, p_flavicirrata_f, s_meleagris_f, l_unguiculata_f,
           l_tetraphylla_f, l_tetraphylla_2_f, a_victoria_f, m_cellularia_f, p_gregarium_f,
           a_aurita_f, c_capillata_f, c_capillata_2_f]  # e_indicans_f

sea_den = 1.024 * np.power(10.0, 6)  # g/m^3, 1.024 g/cm^3 (Colin & Costello, 2001)
sea_vis = np.power(10.0, -6)  # m^2/s


######################################################################
# run my implementation of their model
# param: dfs, constant orifice
######################################################################
def copy_model(df_ref, ori_ref):
    clean_time(df_ref)
    get_basics(df_ref)
    get_dsdt(df_ref, ori_ref)
    return_accel(df_ref)

    # the last row of dSdt are zeroed because it requires the change in time
    # and because it would effect the output we ignore that last row entirely
    df_ref.drop(df_ref.tail(1).index, inplace=True)


######################################################################
# run a tweaked version of their model
# param: dfs
######################################################################
def tweaked_model(df_ref):
    clean_time(df_ref)
    get_basics(df_ref)
    get_dsdt(df_ref)
    return_accel(df_ref, True)

    df_ref.drop(df_ref.tail(1).index, inplace=True)


######################################################################
# run my implementation of their model
# param: dfs, constant orifice, and species names
######################################################################
def improved_model(df_ref):
    clean_time(df_ref)
    get_basics(df_ref)
    get_accel(df_ref)

    df_ref.drop(df_ref.tail(1).index, inplace=True)


######################################################################
# clean up dataframe to show just 1 time reference
# inspect time data in odd columns and determine whether they are reasonably
# equivalent to the column 'st' time reference, and drop the column if so
######################################################################
def clean_time(df_ref):
    to_delete = []
    for column in df_ref.columns:
        index_no = df_ref.columns.get_loc(column)
        if index_no % 2 == 1:
            eq = True
            # assume no more than 2 references in a row are considerably different to the standard time
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
    if to_delete:
        df_ref.drop(to_delete, axis=1, inplace=True)


######################################################################
# add instantaneous heights, diameters, and update velocities and
# accelerations to the corrected units
# param: list of medusae dataframes
######################################################################
def get_basics(df_ref):
    accelerations_o = []
    accelerations_m = []
    velocities = []
    heights = []
    diameters = []

    has_ao = False
    has_a = False
    has_v = False
    for column in df_ref.columns:
        if re.search(r'ao', column):
            has_ao = True
        elif re.search(r'a', column):
            has_a = True
        if re.search(r'v', column):
            has_v = True

    if not has_ao and not has_a and not has_v:
        return

    for row in df_ref.index:
        if has_ao:
            aoc = df_ref.at[row, 'ao'] / 100.0
            accelerations_o.append(aoc)
            amc = df_ref.at[row, 'am'] / 100.0
            accelerations_m.append(amc)
        elif has_a:
            aoc = df_ref.at[row, 'a'] / 100.0
            accelerations_o.append(aoc)
        u = df_ref.at[row, 'v'] / 100.0  # convert velocity unit to m/s
        velocities.append(u)
        d_h = bell_dim(df_ref.at[row, 're'], u, df_ref.at[row, 'f'])  # re: m^2/s / m^2/s, fineness: m/m
        diameters.append(d_h[0])
        heights.append(d_h[1])

    if not has_ao and not has_a:
        for row in (list(range(len(df_ref.index) - 1))):
            t1 = df_ref.at[row, 'st']
            t2 = df_ref.at[row + 1, 'st']
            v1 = df_ref.at[row, 'v']
            v2 = df_ref.at[row + 1, 'v']
            aoc = (v2 - v1) / (t2 - t1)
            accelerations_o.append(aoc)
        accelerations_o.append(0)

    if has_ao:
        df_ref["am"] = accelerations_m
    else:
        df_ref.rename(columns={"a": "ao"}, inplace=True)  # WHY DOESN'T THIS WORK
    df_ref["ao"] = accelerations_o
    df_ref["v"] = velocities
    df_ref["h"] = heights
    df_ref["d"] = diameters


######################################################################
# add thrust and dsdt using acceleration, mass, and drag
######################################################################
def get_dsdt(df_ref, ori_ref=None):
    orifices = []
    volumes = []
    masses = []
    drags = []
    net_forces = []
    thrusts = []
    dsdt = []

    for row in df_ref.index:
        am = df_ref.at[row, 'am']
        v = df_ref.at[row, 'v']
        r = df_ref.at[row, 're']
        h = df_ref.at[row, 'h']
        d = df_ref.at[row, 'd']
        if ori_ref is None:
            orifices.append(ori(d))
        else:
            orifices.append(ori_ref)
        volumes.append(bell_vol(h, d))
        masses.append(bell_mas(h, d))
        net_forces.append(nf_a(h, d, am))
        drags.append(bell_drag(r, h, d, v))
        thrusts.append(tf_a(h, d, am, r, v))

    df_ref["ori"] = orifices
    df_ref["V"] = volumes
    df_ref["m"] = masses
    df_ref["nf"] = net_forces
    df_ref["drg"] = drags
    df_ref["tf"] = thrusts

    for row in (list(range(len(df_ref.index) - 1))):
        o = df_ref.at[row, 'ori']
        v1 = df_ref.at[row, 'V']
        v2 = df_ref.at[row + 1, 'V']
        f = df_ref.at[row, 'tf']
        if v1 > v2:
            dsdt.append(-1 * np.sqrt(o / sea_den * f))
        else:
            dsdt.append(np.sqrt(o / sea_den * f))

    dsdt.append(0)

    df_ref["dSdt"] = dsdt


######################################################################
# add acceleration based on the modeled acceleration estimate.
# used to check if the model has been implemented correctly
# and compare to the observed acceleration
######################################################################
def return_accel(df_ref, improved=False):
    new_thrusts = []
    new_net_forces = []
    accelerations = []

    for row in df_ref.index:
        o = df_ref.at[row, "ori"]
        st = df_ref.at[row, "dSdt"]
        new_thrusts.append(sea_den / o * np.power(st, 2))

    df_ref["tf"] = new_thrusts

    for row in df_ref.index:
        thr = df_ref.at[row, 'tf']
        drag = df_ref.at[row, 'drg']
        if improved is True:
            new_net_forces.append(nf_tf(3*thr, drag))
        else:
            new_net_forces.append(nf_tf(thr, drag))

    df_ref["nf"] = new_net_forces

    for row in df_ref.index:
        m = df_ref.at[row, 'm']
        f = df_ref.at[row, 'nf']
        accelerations.append(f / m)

    df_ref["ac"] = accelerations


######################################################################
# add ds, dv and other measurements of volume changes
# using thrust, volume, and time
######################################################################
def sub_n_vol_change(df_ref, ori_ref):
    thrust = ""
    dv = []
    ds = []
    dvdt = []
    dsdv = []

    for column in df_ref.columns:
        if re.search(r'tf', column):
            thrust = re.search(r'tf', column).string
    for row in (list(range(len(df_ref.index) - 1))):
        t = df_ref.at[0, 'st']
        v1 = df_ref.at[row, 'V']
        v2 = df_ref.at[row + 1, 'V']
        dv.append(v2 - v1)
        dvdt.append((v2 - v1) / t)
        f = df_ref.at[row, thrust]
        sv = dsdv_tf(f, ori_ref, v1, v2, t)
        dsdv.append(sv)
        ds.append(sv * (v2 - v1))

    dv.append(0)
    ds.append(0)
    dvdt.append(0)
    dsdv.append(0)

    df_ref["dV"] = dv
    df_ref["dS"] = ds
    df_ref["dVdt"] = dvdt
    df_ref["dSdV"] = dsdv


######################################################################
# add acceleration using mass, drag and a calculated thrust
######################################################################
def get_accel(df_ref):

    volumes = []
    masses = []
    orifices = []
    drags = []
    thrusts = []
    net_forces = []
    accelerations = []
    dsdt = []

    for row in df_ref.index:
        h = df_ref.at[row, 'h']
        d = df_ref.at[row, 'd']
        r = df_ref.at[row, 're']
        u = df_ref.at[row, 'v']
        orifices.append(ori(d))
        volumes.append(bell_vol(h, d))
        masses.append(bell_mas(h, d))
        drags.append(bell_drag(r, h, d, u))

    df_ref["V"] = volumes
    df_ref["ori"] = orifices

    for row in list(range(len(df_ref.index) - 1)):
        o = df_ref.at[row, 'ori']
        vol1 = df_ref.at[row, 'V']
        vol2 = df_ref.at[row+1, 'V']
        t1 = df_ref.at[row, 'st']
        t2 = df_ref.at[row+1, 'st']
        st = ds_dv(vol2-vol1)/(t2 - t1)
        dsdt.append(st)
        thrusts.append(tf_dsdt(o, st))

    dsdt.append(0)
    thrusts.append(0)

    df_ref["dSdt"] = dsdt
    df_ref["drg"] = drags
    df_ref["tf"] = thrusts

    for row in df_ref.index:
        thr = df_ref.at[row, 'tf']
        drag = df_ref.at[row, 'drg']
        net_forces.append(nf_tf(thr, drag))

    df_ref["m"] = masses
    df_ref["nf"] = net_forces

    for row in df_ref.index:
        m = df_ref.at[row, 'm']
        f = df_ref.at[row, 'nf']
        accelerations.append(f / m)

    df_ref["ac"] = accelerations


######################################################################
# add thrust and change of bell volume(why?)
# by using velocity, bell dimensions, re, and time
######################################################################
def get_thrust(df_ref):
    volumes = []
    thrusts = []
    masses = []
    drags = []
    forces = []
    # dv = []

    for row in df_ref.index:
        v = df_ref.at[row, 'v']
        r = df_ref.at[row, 're']
        h = df_ref.at[row, 'h']
        d = df_ref.at[row, 'd']
        a = df_ref.at[row, 'ao']
        volumes.append(bell_vol(h, d))
        masses.append(bell_mas(h, d))
        drags.append(bell_drag(r, h, d, v))
        forces.append(nf_a(h, d, a))
        thrusts.append(tf_a2(h, d, a, r, v))

    df_ref["V"] = volumes
    df_ref["m"] = masses
    df_ref["drg"] = drags
    df_ref["nf"] = forces
    df_ref["tf"] = thrusts

    # for row in (list(range(len(df_ref.index) - 1))):
    #     t = df_ref.at[row, 'st']
    #     v1 = df_ref.at[row, 'V']
    #     v2 = df_ref.at[row+1, 'V']
    #     dv.append((v2-v1)/t)
    #
    # dv.append(0)
    # df_ref["dV"] = dv


######################################################################
# calculate bell diameter (m) and height (m)
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
# calculate bell volume (m^3)
# param: bell height(m), bell diameter(m)
######################################################################
def bell_vol(h_ref, d_ref):
    # bell radius = bell diameter / 2
    # radius: m
    radius = d_ref / 2
    # bell volume = 2/3 * bell height * pi * radius^2
    # volume: m^3 = m * m^2
    volume = 2 / 3 * h_ref * np.pi * np.power(radius, 2)
    return volume


######################################################################
# calculate effective mass (g)
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
# calculate drag (g * m / s^2)
# param: Re, bell height(m), bell diameter(m), swimming velocity (m/s)
######################################################################
def bell_drag(re_ref, h_ref, d_ref, u_ref):
    # if re_ref > 700:
    #     return 0
    # elif re_ref < 1:
    if re_ref < 1:
        # drag coefficient = 24 / re
        # coefficient:
        coe = 24 / re_ref
    elif re_ref < 500:
        # drag coefficient = 24 / re^0.7
        # coefficient:
        coe = 24 / np.power(re_ref, 0.7)
    else:
        # drag coefficient = 24 / re^0.7
        # coefficient:
        coe = 24 / np.power(re_ref, 0.5)

    # bell surface area = pi * bell height * bell diameter / 4
    # area: m^2 = m * m
    area = np.pi * h_ref * d_ref / 4

    # drag force = sea density * swimming velocity^2 * bell surface area / 2
    # drag: g * m / s^2 = g/m^3 * (m/s)^2 * m^2
    drag = (sea_den * np.power(u_ref, 2) * area * coe) / 2

    if drag < 0.000001:
        print("SMALL DRAG WARNING: coe %f area %f veloc %f" % (coe, area, u_ref))

    return drag


######################################################################
# calculate orifice area (m^2)
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
# get bell fineness
# param: bell height(m), bell diameter(m)
######################################################################
def fin(h_ref, d_ref):
    # bell fineness = bell height / bell diameter
    # fineness: m/m
    fineness = h_ref / d_ref
    return fineness


######################################################################
# get net force (g * m / s^2) from acceleration
# param: bell height(m), bell diameter(m), modeled acceleration(m/s^2)
######################################################################
def nf_a(h_ref, d_ref, a_ref):
    mass = bell_mas(h_ref, d_ref)
    # force = mass * acceleration
    # force: g * m / s^2 = g * (m / s^2)
    net_force = mass * a_ref
    return net_force


######################################################################
# get thrust (g * m / s^2) from acceleration
# param: bell height(m), bell diameter(m), acceleration(m/s^2),
#        Re, swimming velocity (m/s)
######################################################################
def tf_a(h_ref, d_ref, a_ref, re_ref, u_ref):
    force = nf_a(h_ref, d_ref, a_ref)
    drag = bell_drag(re_ref, h_ref, d_ref, u_ref)
    thrust = force + drag

    if thrust < 0:
        return 0
    else:
        return thrust


######################################################################
# get thrust (g * m / s^2) from acceleration
# param: bell height(m), bell diameter(m), acceleration(m/s^2),
#        Re, swimming velocity (m/s)
######################################################################
def tf_a2(h_ref, d_ref, a_ref, re_ref, u_ref):
    force = nf_a(h_ref, d_ref, a_ref)
    drag = bell_drag(re_ref, h_ref, d_ref, u_ref)
    thrust = force + drag
    return thrust


######################################################################
# get net force (g * m / s^2) from thrust and drag
# param: thrust (g * m / s^2), drag (g * m / s^2)
######################################################################
def nf_tf(thr_ref, drg_ref):
    # force = thrust - drag
    # force: g * m / s^2 = g * m / s^2 - g * m / s^2
    net_force = thr_ref - drg_ref
    return net_force


######################################################################
# get acceleration (m / s^2) from thrust, drag, and mass
# param: thrust (g * m / s^2), drag (g * m / s^2), mass (g)
######################################################################
def a_tf(thr_ref, drg_ref, mas_ref):
    # acceleration = (thrust - drag) / mass
    # acceleration: (g * m / s^2 - g * m / s^2) / g
    acc = (thr_ref - drg_ref) / mas_ref
    return acc


######################################################################
# change of subumbrellar volume with respect to the change of
# bell volume
######################################################################
def dsdv_tf(thrust_ref, ori_ref, vol_1, vol_2, t):
    # this calculated dsdt value will always be positive, but it is
    # negative when dv when dvdt is negative, which notes a contracting
    # bell. therefore, dsdv should always be positive
    dsdt = np.sqrt(ori_ref / sea_den * thrust_ref)
    dvdt = (vol_2 - vol_1)/t
    dsdv = np.absolute(dsdt/dvdt)
    return dsdv


######################################################################
# estimate change of prolate and oblate subumbrellar volume
# equation acquired from Sub_volume.py
######################################################################
def ds_dv(dv_ref, oblate=True):
    if oblate:
        ds = 3*(0.345 * dv_ref - 1.34e-07)
    else:
        ds = 1.263 * dv_ref + 7.357e-09

    return ds


######################################################################
# get thrust (g * m / s^2) from dSdt
# param: orifice area, dSdt
######################################################################
def tf_dsdt(ori_ref, dsdt_ref):
    # g * m / (s^2) = (g / m^3) / (m^2) * (m^3/s)^2
    thrust = sea_den / ori_ref * np.power(dsdt_ref, 2)
    return thrust


######################################################################
# split p_gregarium dataframe into 4 based on volume
# param: df
######################################################################
def split_digitale(df_ref):
    ref_1 = df_ref[3:19].copy()
    ref_1['st'] = np.arange(0, df_ref.at[13, 'st'] + 0.04, df_ref.at[13, 'st'] / 15)
    ref_2 = df_ref[19:33].copy()
    ref_2['st'] = np.arange(0, df_ref.at[13, 'st'] + 0.05, df_ref.at[13, 'st'] / 13)
    ref_3 = df_ref[33:45].copy()
    ref_3['st'] = np.arange(0, df_ref.at[13, 'st'] + 0.05, df_ref.at[13, 'st'] / 11)

    return [ref_1, ref_2, ref_3]


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
def split_flavicirrata(df_ref):
    ref_1 = df_ref[5:15].copy()
    ref_1['st'] = np.arange(0, df_ref.at[9, 'st'] + 0.05, df_ref.at[9, 'st'] / 9)
    ref_2 = df_ref[19:30].copy()
    ref_2['st'] = np.arange(0, df_ref.at[9, 'st'] + 0.05, df_ref.at[9, 'st'] / 10)

    return [ref_1, ref_2]


######################################################################
# split m_cellularia dataframe into 4 based on volume
# param: df
######################################################################
def split_victoria(df_ref):
    ref_1 = df_ref[2:11].copy()
    ref_1['st'] = np.arange(0, df_ref.at[7, 'st'] + 0.05, df_ref.at[7, 'st'] / 8)
    ref_2 = df_ref[11:18].copy()
    ref_2['st'] = np.arange(0, df_ref.at[7, 'st'] + 0.05, df_ref.at[7, 'st'] / 6)
    ref_3 = df_ref[18:26].copy()
    ref_3['st'] = np.arange(0, df_ref.at[7, 'st'] + 0.05, df_ref.at[7, 'st'] / 7)
    ref_4 = df_ref[26:32].copy()
    ref_4['st'] = np.arange(0, df_ref.at[7, 'st'] + 0.05, df_ref.at[7, 'st'] / 5)
    return [ref_1, ref_2, ref_3, ref_4]


######################################################################
# split m_cellularia dataframe into 4 based on volume
# param: df
######################################################################
def split_cellularia(df_ref):
    ref_1 = df_ref[4:15].copy()
    ref_1['st'] = np.arange(0, df_ref.at[11, 'st'] + 0.05, df_ref.at[11, 'st'] / 10)
    ref_2 = df_ref[15:27].copy()
    ref_2['st'] = np.arange(0, df_ref.at[11, 'st'] + 0.05, df_ref.at[11, 'st'] / 11)
    ref_3 = df_ref[27:41].copy()
    ref_3['st'] = np.arange(0, df_ref.at[11, 'st'] + 0.05, df_ref.at[11, 'st'] / 13)
    ref_4 = df_ref[41:54].copy()
    ref_4['st'] = np.arange(0, df_ref.at[11, 'st'] + 0.05, df_ref.at[11, 'st'] / 12)
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
# split a_aurita dataframe into 4 based on volume
# param: df
######################################################################
def split_aurita(df_ref):
    ref_1 = df_ref[3:11].copy()
    ref_1['st'] = np.arange(0, df_ref.at[7, 'st'] + 0.05, df_ref.at[7, 'st'] / 7)
    ref_2 = df_ref[11:19].copy()
    ref_2['st'] = np.arange(0, df_ref.at[7, 'st'] + 0.05, df_ref.at[7, 'st'] / 7)
    ref_3 = df_ref[19:27].copy()
    ref_3['st'] = np.arange(0, df_ref.at[7, 'st'] + 0.05, df_ref.at[7, 'st'] / 7)
    ref_4 = df_ref[27:35].copy()
    ref_4['st'] = np.arange(0, df_ref.at[7, 'st'] + 0.05, df_ref.at[7, 'st'] / 7)

    return [ref_1, ref_2, ref_3, ref_4]


######################################################################
# split s_meleagris dataframe into 4 based on volume
# param: df
######################################################################
def split_meleagris(df_ref):
    ref_1 = df_ref[5:16].copy()
    ref_1['st'] = np.arange(0, df_ref.at[10, 'st'] + 0.005, df_ref.at[10, 'st'] / 10)
    ref_2 = df_ref[16:26].copy()
    ref_2['st'] = np.arange(0, df_ref.at[10, 'st'] + 0.005, df_ref.at[10, 'st'] / 9)
    ref_3 = df_ref[26:37].copy()
    ref_3['st'] = np.arange(0, df_ref.at[10, 'st'] + 0.005, df_ref.at[10, 'st'] / 10)
    ref_4 = df_ref[37:48].copy()
    ref_4['st'] = np.arange(0, df_ref.at[10, 'st'] + 0.005, df_ref.at[10, 'st'] / 10)

    return [ref_1, ref_2, ref_3, ref_4]


######################################################################
# split l_unguiculata dataframe into 5 based on volume
# param: df
######################################################################
def split_unguiculata(df_ref):
    ref_1 = df_ref[2:12].copy()
    ref_1['st'] = np.arange(0, df_ref.at[9, 'st'] + 0.005, df_ref.at[9, 'st'] / 9)
    ref_2 = df_ref[12:22].copy()
    ref_2['st'] = np.arange(0, df_ref.at[9, 'st'] + 0.005, df_ref.at[9, 'st'] / 9)
    ref_3 = df_ref[22:33].copy()
    ref_3['st'] = np.arange(0, df_ref.at[9, 'st'] + 0.005, df_ref.at[9, 'st'] / 10)
    ref_4 = df_ref[33:43].copy()
    ref_4['st'] = np.arange(0, df_ref.at[9, 'st'] + 0.005, df_ref.at[9, 'st'] / 9)
    ref_5 = df_ref[43:53].copy()
    ref_5['st'] = np.arange(0, df_ref.at[9, 'st'] + 0.005, df_ref.at[9, 'st'] / 9)

    return [ref_1, ref_2, ref_3, ref_4, ref_5]


######################################################################
# split c_capillata dataframe into 4 based on volume
# param: df
######################################################################
def split_capillata(df_ref):
    ref_1 = df_ref[3:19].copy()
    ref_1['st'] = np.arange(0, df_ref.at[14, 'st'] + 0.005, df_ref.at[14, 'st'] / 15)
    ref_2 = df_ref[19:33].copy()
    ref_2['st'] = np.arange(0, df_ref.at[14, 'st'] + 0.005, df_ref.at[14, 'st'] / 13)
    ref_3 = df_ref[33:50].copy()
    ref_3['st'] = np.arange(0, df_ref.at[14, 'st'] + 0.005, df_ref.at[14, 'st'] / 16)
    ref_4 = df_ref[50:64].copy()
    ref_4['st'] = np.arange(0, df_ref.at[14, 'st'] + 0.005, df_ref.at[14, 'st'] / 13)

    return [ref_1, ref_2, ref_3, ref_4]


######################################################################
# split c_capillata_2 dataframe into 3 based on volume
# param: df
######################################################################
def split_capillata2(df_ref):
    ref_1 = df_ref[6:34].copy()
    ref_1['st'] = np.arange(0, df_ref.at[25, 'st'] + 0.005, df_ref.at[25, 'st'] / 27)
    ref_2 = df_ref[34:58].copy()
    ref_2['st'] = np.arange(0, df_ref.at[25, 'st'] + 0.005, df_ref.at[25, 'st'] / 23)
    ref_3 = df_ref[58:86].copy()
    ref_3['st'] = np.arange(0, df_ref.at[25, 'st'] + 0.005, df_ref.at[25, 'st'] / 27)

    return [ref_1, ref_2, ref_3]


######################################################################
# split l_tetraphylla dataframe into 2 based on volume
# param: df
######################################################################
def split_tetraphylla(df_ref):
    ref_1 = df_ref[11:34].copy()
    ref_1['st'] = np.arange(0, df_ref.at[24, 'st'] + 0.0005, df_ref.at[24, 'st'] / 22)
    ref_2 = df_ref[34:62].copy()
    ref_2['st'] = np.arange(0, df_ref.at[24, 'st'] + 0.0005, df_ref.at[24, 'st'] / 27)

    return [ref_1, ref_2]


######################################################################
# split l_tetraphylla_2 dataframe into 2 based on volume
# param: df
######################################################################
def split_tetraphylla2(df_ref):
    ref_1 = df_ref[11:35].copy()
    ref_1['st'] = np.arange(0, df_ref.at[25, 'st'] + 0.0005, df_ref.at[25, 'st'] / 23)
    ref_2 = df_ref[35:64].copy()
    ref_2['st'] = np.arange(0, df_ref.at[25, 'st'] + 0.0005, df_ref.at[25, 'st'] / 28)

    return [ref_1, ref_2]