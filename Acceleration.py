import pandas as pd
import numpy as np
import re
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# global variables

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
# task 2: model accurate oblate medusae acceleration
######################################################################
def main():
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

    # oris= [(ori(0.83 / 100)), (ori(0.85 / 100)), (ori(0.56 / 100)),
    #              (ori(5 / 100)), (ori(6.5 / 100)), (ori(2.14 / 100))]

    improved_model(dfs, name)

    dfss = [split_victoria(dfs[3]), split_cellularia(dfs[4]), split_gregarium(dfs[5])]


    medusae = ['a.victoria', 'm.cellularia', 'p.gregarium']
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
                # print("row %i value %f" % (row, df.loc[row, 'am']))
                if df.loc[row, 'ac'] > 100:
                    # print("row %i value %f" % (row, df.loc[row, 'am']))
                    df.drop(row, inplace=True)

    dfs_count = 0
    for dfs in dfss:
        total_x = np.array([])
        total_y = np.array([])
        colors = ['goldenrod', 'firebrick', 'forestgreen', 'dodgerblue']
        df_count = 0
        for df in dfs:
            label = "cycle #" + str(df_count + 1)
            plt.plot(df["st"], df["tm"], color=colors[df_count], label=label)
            total_x = np.append(total_x, df["st"].values)
            total_y = np.append(total_y, df["tm"].values)
            df_count += 1
        input_regression = [('polynomial', PolynomialFeatures(degree=2)), ('modal', LinearRegression())]
        pipe = Pipeline(input_regression)
        pipe.fit(total_x.reshape(-1, 1), total_y.reshape(-1, 1))
        poly_pred = pipe.predict(total_x.reshape(-1, 1))
        sorted_zip = sorted(zip(total_x, poly_pred))
        x_poly, poly_pred = zip(*sorted_zip)
        reg_label = ('RMSE %f' % np.sqrt(mean_squared_error(x_poly, poly_pred)))
        plt.plot(x_poly, poly_pred, color='magenta', linestyle='--', label=reg_label)
        plt.title("%s thrust over time" % (medusae[dfs_count]))
        plt.legend()
        plt.xlabel("time s")
        plt.ylabel("thrust g*m*s^-2")
        plt.tight_layout()
        plt.show()
        dfs_count += 1

    dfs_count = 0
    for dfs in dfss:
        total_x = np.array([])
        total_y = np.array([])
        colors = ['goldenrod', 'firebrick', 'forestgreen', 'dodgerblue']
        df_count = 0
        for df in dfs:
            label = "cycle #" + str(df_count + 1)
            plt.plot(df["st"], df["ac"], color=colors[df_count], label=label)
            total_x = np.append(total_x, df["st"].values)
            total_y = np.append(total_y, df["ac"].values)
            df_count += 1
        input_regression = [('polynomial', PolynomialFeatures(degree=2)), ('modal', LinearRegression())]
        pipe = Pipeline(input_regression)
        pipe.fit(total_x.reshape(-1, 1), total_y.reshape(-1, 1))
        poly_pred = pipe.predict(total_x.reshape(-1, 1))
        sorted_zip = sorted(zip(total_x, poly_pred))
        x_poly, poly_pred = zip(*sorted_zip)
        reg_label = ('RMSE %f' % np.sqrt(mean_squared_error(x_poly, poly_pred)))
        plt.plot(x_poly, poly_pred, color='magenta', linestyle='--', label=reg_label)
        plt.title("%s acceleration over time" % (medusae[dfs_count]))
        plt.legend()
        plt.xlabel("time s")
        plt.ylabel("acceleration ms^-2")
        plt.tight_layout()
        plt.show()
        dfs_count += 1

    total_x = np.array([])
    total_y1 = np.array([])
    total_y2 = np.array([])
    count = 0
    for df in dfs:
        label = "cycle #" + str(count + 1)
        plt.plot(df["st"], df["ac"], color='magenta', label=label)
        plt.plot(df["st"], df["am"], color='purple', label=label)
        total_x = np.append(total_x, df["st"].values)
        total_y1 = np.append(total_y1, df["ac"].values)
        total_y2 = np.append(total_y2, df["am"].values)
        count += 1
    input_regression = [('polynomial', PolynomialFeatures(degree=2)), ('modal', LinearRegression())]
    pipe1 = Pipeline(input_regression)
    pipe1.fit(total_x.reshape(-1, 1), total_y1.reshape(-1, 1))
    poly_pred1 = pipe1.predict(total_x.reshape(-1, 1))
    sorted_zip1 = sorted(zip(total_x, poly_pred1))
    x_poly, poly_pred1 = zip(*sorted_zip1)
    reg_label1 = ('RMSE %f' % np.sqrt(mean_squared_error(x_poly, poly_pred1)))
    plt.plot(x_poly, poly_pred1, color='magenta', linestyle='--', label=reg_label1)
    pipe2 = Pipeline(input_regression)
    pipe2.fit(total_x.reshape(-1, 1), total_y2.reshape(-1, 1))
    poly_pred2 = pipe2.predict(total_x.reshape(-1, 1))
    sorted_zip2 = sorted(zip(total_x, poly_pred2))
    x_poly, poly_pred2 = zip(*sorted_zip2)
    reg_label2 = ('RMSE %f' % np.sqrt(mean_squared_error(x_poly, poly_pred1)))
    plt.plot(x_poly, poly_pred2, color='purple', linestyle='--', label=reg_label2)
    plt.title("p_gregarium acceleration over time")
    plt.legend()
    plt.xlabel("time s")
    plt.ylabel("acceleration ms^-2")
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
# run my implementation of their model
# param: dfs, constant orifice, and species names
######################################################################
def improved_model(df_ref, name_ref):
    count = 0
    for (df, name) in zip(df_ref, name_ref):
        clean_time(df)
        basics(df)
        get_accel(df)
        df.drop(df.tail(1).index, inplace=True)
        plt.plot(df["st"], df["ac"], label='modeled acceleration')
        print("count" + str(df))
        if name == 's_sp' or name == 'p_gregarium':
            plt.plot(df["st"], df["am"], label='published acceleration')
            plt.plot(df["st"], df["ao"], label='observed acceleration')
        else:
            plt.plot(df["st"], df["a"], label='observed acceleration')
        plt.title("modeled %s acceleration over 4 cycles" % name)
        plt.legend()
        plt.xlabel("time s")
        plt.ylabel("acceleration m/s^2")
        plt.tight_layout()
        plt.show()

        count += 1

######################################################################
# clean up data frame to show only 1 corrected time reference
# param: list of medusae dataframes
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
# add instantaneous heights, diameters, and velocities and accelerations
# in the corrected units. these complete the basic measurements
# param: list of medusae dataframes
######################################################################
def basics(df_ref):
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
    accelerations_o = []  # store instantaneous accelerations in converted units
    accelerations_m = []  # store instantaneous accelerations in converted units
    velocities = []  # store instantaneous velocities in converted units
    heights = []  # store instantaneous heights
    diameters = []  # store instantaneous diameters

    has_am = False
    for column in df_ref.columns:
        if re.search(r'am', column):
            has_am = True

    for row in df_ref.index:
        # aoc = df_ref.at[row, 'a'] / 100.0
        # aoc_col = df_ref.columns.get_loc('a')
        # aoc = df_ref.iat[aoc_col, row] / 100.0
        #
        # accelerations_o.append(aoc)
        if has_am:
            aoc = df_ref.at[row, 'ao'] / 100.0
            accelerations_o.append(aoc)
            amc = df_ref.at[row, 'am'] / 100.0
            accelerations_m.append(amc)
        else:
            aoc = df_ref.at[row, 'a'] / 100.0
            accelerations_o.append(aoc)
        u = df_ref.at[row, 'v'] / 100.0  # convert velocity unit to m/s
        velocities.append(u)
        d_h = dim(df_ref.at[row, 're'], u, df_ref.at[row, 'f'])  # re: m^2/s / m^2/s, fineness: m/m
        diameters.append(d_h[0])
        heights.append(d_h[1])


    if has_am:
        df_ref["ao"] = accelerations_o
        df_ref["am"] = accelerations_m
    else:
        df_ref["a"] = accelerations_o
    df_ref["v"] = velocities
    df_ref["h"] = heights
    df_ref["d"] = diameters


######################################################################
# find the modeled thrust force based on the modeled acceleration
# derived by the basic measurements
######################################################################
def get_accel(df_ref):

    # volumes = []  # store instantaneous volumes
    # masses = []  # store instantaneous masses
    # drags = []  # store instantaneous drags
    # dSdt = []  # store instantaneous change in subumbrellar volume over change of time
    # thrusts = []  # store instantaneous thrust force
    # net_forces = []  # store instantaneous total force
    # accelerations = []  # store instantaneous acceleration
    #
    # for row in df_ref.index:
    #     h = df_ref.at[row, 'h']
    #     d = df_ref.at[row, 'd']
    #     r = df_ref.at[row, 're']
    #     u = df_ref.at[row, 'v']
    #     volumes.append(vol(h, d))
    #     masses.append(mas(h, d))
    #     drags.append(drg(r, h, d, u))
    #
    # df_ref["V"] = volumes
    # df_ref["ori"] = ori_ref
    #
    # for row in list(range(len(df_ref.index) - 1)):
    #     v1 = df_ref.at[row, 'V']
    #     v2 = df_ref.at[row + 1, 'V']
    #     t1 = df_ref.at[row, 'st']
    #     t2 = df_ref.at[row + 1, 'st']
    #     o = df_ref.at[row, 'ori']
    #     dSdt.append(dS(v2 - v1) / (t2 - t1))
    #     thrusts.append(thr_p(o, v1, v2, t1, t2))
    #
    # dSdt.append(0)
    # thrusts.append(0)
    #
    # df_ref["drg"] = drags
    # df_ref["dSdt"] = dSdt
    # df_ref["tm"] = thrusts
    #
    # for row in df_ref.index:
    #     thr = df_ref.at[row, 'tm']
    #     drag = df_ref.at[row, 'drg']
    #     net_forces.append(nfr(thr, drag))
    #
    # df_ref["m"] = masses
    # df_ref["nfm"] = net_forces
    #
    # for row in df_ref.index:
    #     m = df_ref.at[row, 'm']
    #     f = df_ref.at[row, 'nfm']
    #     accelerations.append(f / m)
    #
    # df_ref["ac"] = accelerations
    volumes = []  # store instantaneous volumes
    masses = []  # store instantaneous masses
    orifices = []
    drags = []  # store instantaneous drags
    thrusts = []  # store instantaneous thrust force
    net_forces = []  # store instantaneous total force
    accelerations = []  # store instantaneous acceleration
    dSdt = []

    for row in df_ref.index:
        h = df_ref.at[row, 'h']
        d = df_ref.at[row, 'd']
        r = df_ref.at[row, 're']
        u = df_ref.at[row, 'v']
        orifices.append(ori(d))
        volumes.append(vol(h, d))
        masses.append(mas(h, d))
        drags.append(drg(r, h, d, u))

    df_ref["V"] = volumes
    df_ref["ori"] = orifices

    for row in list(range(len(df_ref.index) - 1)):
        o = df_ref.at[row, 'ori']
        vol1 = df_ref.at[row, 'V']
        vol2 = df_ref.at[row+1, 'V']
        t1 = df_ref.at[row, 'st']
        t2 = df_ref.at[row+1, 'st']
        dsdt = dS(vol2-vol1)/(t2 - t1)
        dSdt.append(dsdt)
        thrusts.append(3*(sea_den / o * np.power(dsdt, 2)))

    dSdt.append(0)
    thrusts.append(0)

    df_ref["dSdt"] = dSdt
    df_ref["drg"] = drags
    df_ref["tm"] = thrusts

    for row in df_ref.index:
        thr = df_ref.at[row, 'tm']
        drag = df_ref.at[row, 'drg']
        net_forces.append(nfr(thr, drag))


    df_ref["m"] = masses
    df_ref["nfm"] = net_forces

    for row in df_ref.index:
        m = df_ref.at[row, 'm']
        f = df_ref.at[row, 'nfm']
        accelerations.append(f / m)

    df_ref["ac"] = accelerations


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
# estimate change of oblate subumbrellar volume
######################################################################
def dS(dV):
    # formula acquired from Sub_volume.py
    ds = 1.263 * dV + 7.357e-09
    return ds


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
