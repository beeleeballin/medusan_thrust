import pandas as pd
import numpy as np

# global variables

f_path = "/Users/beelee/Desktop/10-07-data.csv"

vel_o = 0
acc_m = 0
acc_o = 0

h_b = 0.0
d_b = 0.0
s_b = 0.0

t_rel = 0
t_pau = 0
t_con = 0
t_dri = 0


sea_den = 1.024 * np.power(10.0, 6)  # 1/m^3, 1.024 1/cm^3 (Colin & Costello, 2001)
sea_vis = np.power(10.0, -6)  # m^2/s

F_s = 0

acc_cor = 0


def main():
    # import csv into 2 dataframes, 1 prolate 1 oblate
    df = pd.read_csv(f_path)
    dfs = np.split(df, [26], axis=1)  # may need a better system splitting

    # dfs[0].drop(list(range(41, 46)), axis=0, inplace=True)
    dfs[0].dropna(axis=1, how='all', inplace=True)
    dfs[1].drop(list(range(36, 46)), axis=0, inplace=True)
    dfs[1].drop("Unnamed: 27", axis=1, inplace=True)
    dfs[1].dropna(axis=1, how='all', inplace=True)
    # print(dfs[0].shape, dfs[1].shape)
    print(dfs[:])

    time_pro = {'time': np.arange(0.05, 2.30, 0.05)}
    a_digitale = pd.DataFrame(time_pro, columns=['time', 'height', 'diameter'])

    for time in list(range(len(dfs[0].index)-1)):
        u = dfs[0].loc[time + 1].v / 100.0  # m^2/s
        dim(dfs[0].loc[time].re, u, dfs[0].loc[time].f)  # re: m^2/s / m^2/s, fineness: m/m
        # print('height %f and diameter %f at %f' % (h_b, d_b, (time + 1) * 0.05))
        a_digitale.loc[a_digitale.index[time], 'height'] = h_b
        a_digitale.loc[a_digitale.index[time], 'diameter'] = d_b

    # time_obl = {'time': np.arange(1 / 6, 37 / 6, 1 / 6)}
    # a_victoria = pd.DataFrame(time_obl, columns=['time', 'height', 'diameter'])
    # for time in list(range(len(dfs[1].index)-1)):
    #     u = dfs[1].loc[time].'v.3' / 100.0  # m^2/s
    #     print(u)
    #     dim(dfs[1].loc[time].'re.3', u, dfs[1].loc[time].'f.3')  # re: m^2/s / m^2/s, fineness: m/m
    #     print('height %f and diameter %f at %i' % (h_b, d_b, time))
    #     a_victoria.loc[a_digitale.index[time], 'height'] = h_b
    #     a_victoria.loc[a_digitale.index[time], 'diameter'] = d_b
    #
    # # print(a_digitale)
    print(a_digitale)




######################################################################
# set bell diameter and height
# param: Re, fineness
######################################################################
def dim(re_ref, u_ref, f_ref):
    global d_b
    d_b = 1.0 * re_ref * sea_vis / u_ref  # diameter: m
    global h_b
    h_b = d_b * f_ref  # height: m


######################################################################
# get bell volume
# param: bell height, bell diameter
######################################################################
def vol(h_ref, d_ref):
    radius = d_ref / 2
    volume = 2/3 * h_ref * np.power(radius, 2) * np.pi
    return volume


######################################################################
# get bell fineness
# param: bell height, bell diameter
######################################################################
def fin(h_ref, d_ref):
    fineness = h_ref / d_ref
    return fineness


######################################################################
# get effective mass
# param: bell height, bell diameter
######################################################################
def mas(h_ref, d_ref):
    coe = d_ref / (2 * np.power(h_ref, 1.4))
    volume = vol(h_ref, d_ref)
    mass = sea_den * volume * (1 + coe)
    return mass


######################################################################
# get orifice area
# param: bell diameter
######################################################################
def ori(d_ref):
    radius = d_ref / 2
    area = np.power(radius, 2) * np.pi
    return area


######################################################################
# get drag
# param: bell height, bell diameter
######################################################################
def drg(h_ref, d_ref, u_ref):
    re = d_ref * u_ref / sea_vis
    area = np.pi * h_ref * d_ref / 4
    coe = 0
    if re < 1:
        coe = 24 / re
    elif re < 500:
        coe = 24 / np.power(re, 0.7)
    if coe == 0:
        return 0
    else:
        return (sea_den * np.power(u_ref, 2) * area * coe)/2


######################################################################
# get modeled net force
######################################################################
def nfr_m(h_ref, d_ref):
    mass = mas(h_ref, d_ref)
    return acc_m * mass


######################################################################
# modeled thrust
######################################################################
def thr_m(h_ref, d_ref, u_ref):
    force = nfr_m(h_ref, d_ref)
    drag = drg(h_ref, d_ref, u_ref)
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
