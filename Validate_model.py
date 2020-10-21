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
s_sp_f = f_path + "s_sp_data.csv"
p_gregarium_f = f_path + "p_gregarium_data.csv"

sea_den = 1.024 * np.power(10.0, 6)  # g/m^3, 1.024 g/cm^3 (Colin & Costello, 2001)
sea_vis = np.power(10.0, -6)  # m^2/s


######################################################################
# task 2: validate my replicated of the model
######################################################################
def main():
    # import pre-cleaned up csv files, 1 prolate 1 oblates
    s_sp = pd.read_csv(s_sp_f)
    p_gregarium = pd.read_csv(p_gregarium_f)
    # group the two medusae of interest for easy access
    dfs = [s_sp, p_gregarium]
    name = ['s_sp', 'p_gregarium']

    # s_sp: 0.85cm diameter, p_gregarium: 2.14cm diameter (Colin & Costello, 2001)
    s_sp_ori = ori(0.85 / 100)
    p_gregarium_ori = ori(2.14 / 100)
    # group the two constant orifice areas for easy access
    oris = [s_sp_ori, p_gregarium_ori]

    copy_model(dfs, oris, name)
    dfss = [split_sp(dfs[0]), split_gregarium(dfs[1])]

    dfs_count = 0
    for dfs in dfss:
        total_x = np.array([])
        total_y = np.array([])

        df_count = 0
        for df in dfs:
            label = " #" + str(df_count + 1)
            # plt.plot(df["st"], df["ao"], color='goldenrod', label=("observed"+label))
            plt.plot(df["st"], df["am"], color='firebrick', label=("published"+label))
            plt.plot(df["st"], df["ac"], color='forestgreen', label=("modeled"+label))
            # total_x = np.append(total_x, df["st"].values)
            # total_y = np.append(total_y, df["tm"].values)
            df_count += 1
        # polymodel3 = np.poly1d(np.polyfit(total_x, total_y, 7))
        # polyline3 = np.linspace(min(total_x), max(total_x), 100)
        # plt.scatter(total_x, total_y)
        # plt.plot(polyline3, polymodel3(polyline3), label='thrust regression')
        plt.title("modeled acceleration of %s over time" % (name[dfs_count]))
        plt.legend()
        plt.xlabel("time s")
        plt.ylabel("acceleration m * s^-2")
        plt.tight_layout()
        plt.show()

        dfs_count += 1

    dfs_count = 0
    for dfs in dfss:
        acc_per_cycle = np.empty([4, 3])
        df_count = 0
        for df in dfs:
            acc_per_cycle[df_count][0] = sum(df["am"])
            acc_per_cycle[df_count][1] = sum(df["ac"])
            acc_per_cycle[df_count][2] = sum(df["ao"])
            # print(("am %f, ac %f, ao %f") % (sum(df["am"]), sum(df["ac"]), sum(df["ao"])))
            df_count += 1
        labels = ['published', 'corrected', 'observed']
        plt.xlabel('acceleration references')
        plt.ylabel('acceleration per cycle')
        plt.title('acceleration per cycle for %s' % name[dfs_count])
        plt.xticks(range(3), labels)
        width = 0.2
        plt.bar(np.arange(3), acc_per_cycle[0], width=width)
        plt.bar(np.arange(3) + width, acc_per_cycle[1], width=width)
        plt.bar(np.arange(3) + 2*width, acc_per_cycle[2], width=width)
        plt.bar(np.arange(3) + 3*width, acc_per_cycle[3], width=width)
        plt.show()

        dfs_count += 1

    # reimport a pre-cleaned up oblates csv file
    p_gregarium = pd.read_csv(p_gregarium_f)
    # group the two medusae of interest for easy access
    improved_model(p_gregarium)
    dfs = split_gregarium(p_gregarium)

    total_x = np.array([])
    total_y = np.array([])

    df_count = 0
    for df in dfs:
        label = " #" + str(df_count + 1)
        plt.plot(df["st"], df["ao"], color='goldenrod', label=("observed"+label))
        plt.plot(df["st"], df["am"], color='firebrick', label=("published"+label))
        plt.plot(df["st"], df["ac"], color='forestgreen', label=("corrected"+label))
        # total_x = np.append(total_x, df["st"].values)
        # total_y = np.append(total_y, df["tm"].values)
        df_count += 1
    # polymodel3 = np.poly1d(np.polyfit(total_x, total_y, 7))
    # polyline3 = np.linspace(min(total_x), max(total_x), 100)
    # plt.scatter(total_x, total_y)
    # plt.plot(polyline3, polymodel3(polyline3), label='thrust regression')
    plt.title("modeled acceleration of p_gregarium over time")
    plt.legend()
    plt.xlabel("time s")
    plt.ylabel("acceleration m * s^-2")
    plt.tight_layout()
    plt.show()

    acc_per_cycle = np.empty([4, 3])
    print(acc_per_cycle)
    df_count = 0
    for df in dfs:
        acc_per_cycle[df_count][0] = sum(df["am"])
        acc_per_cycle[df_count][1] = sum(df["ac"])
        acc_per_cycle[df_count][2] = sum(df["ao"])
        # print(("am %f, ac %f, ao %f") % (sum(df["am"]), sum(df["ac"]), sum(df["ao"])))
        df_count += 1
    labels = ['published', 'corrected', 'observed']
    plt.xlabel('acceleration references')
    plt.ylabel('acceleration per cycle')
    plt.title('acceleration per cycle for p_gregarium')
    plt.xticks(range(3), labels)
    width = 0.2
    plt.bar(np.arange(3), acc_per_cycle[0], width=width)
    plt.bar(np.arange(3)+width, acc_per_cycle[1], width=width)
    plt.bar(np.arange(3) + 2*width, acc_per_cycle[2], width=width)
    plt.bar(np.arange(3) + 3*width, acc_per_cycle[3], width=width)
    plt.show()


######################################################################
# run my implementation of their model
# param: dfs, constant orifice, and species names
######################################################################
def copy_model(dfs_ref, oris_ref, name_ref):
    count = 0
    for (df, o) in zip(dfs_ref, oris_ref):
        clean_time(df)
        basics(df)
        thrust_model(df, o)
        get_accel(df, o)
        df.drop(df.tail(1).index, inplace=True)
        plt.plot(df["st"], df["ac"], label='modeled acceleration')
        plt.plot(df["st"], df["am"], label='published acceleration')
        # plt.plot(df["st"], df["ao"], label='observed acceleration')
        plt.title("%s acceleration over time" % name_ref[count])
        plt.legend()
        plt.xlabel("time s")
        plt.ylabel("acceleration m/s^2")
        plt.tight_layout()
        plt.show()
        count += 1


######################################################################
# run my implementation of their model
# param: dfs, constant orifice, and species names
######################################################################
def improved_model(df):

    clean_time(df)
    basics(df)
    correct_thrust_model(df)
    get_correct_accel(df)
    df.drop(df.tail(1).index, inplace=True)


    plt.plot(df["st"], df["ac"], label='modeled acceleration')
    # plt.plot(df["st"], df["am"], label='published acceleration')
    plt.plot(df["st"], df["ao"], label='observed acceleration')
    plt.title("p_gregarium acceleration over time")
    plt.legend()
    plt.xlabel("time s")
    plt.ylabel("acceleration m/s^2")
    plt.tight_layout()
    plt.show()


######################################################################
# split s_sp dataframe into 4 based on volume
# param: df
######################################################################
def split_sp(df_ref):
    ref_1 = df_ref[1:13].copy()
    ref_1["st"] = ref_1["st"] - (df_ref["st"][0] + df_ref["st"][1]) / 2
    ref_2 = df_ref[12:25].copy()
    ref_2["st"] = ref_2["st"] - df_ref["st"][12]
    ref_3 = df_ref[24:36].copy()
    ref_3["st"] = ref_3["st"] - (df_ref["st"][23] + df_ref["st"][24]) / 2
    ref_4 = df_ref[35:45].copy()
    ref_4["st"] = ref_4["st"] - (df_ref["st"][33] + df_ref["st"][34]) / 2
    return [ref_1, ref_2, ref_3, ref_4]


######################################################################
# split p_gregarium dataframe into 4 based on volume
# param: df
######################################################################
def split_gregarium(df_ref):
    ref_1 = df_ref[1:6].copy()
    ref_1["st"] = ref_1["st"] - df_ref["st"][1]
    ref_2 = df_ref[5:9].copy()
    ref_2["st"] = ref_2["st"] - (df_ref["st"][4] + df_ref["st"][5]) / 2
    ref_3 = df_ref[8:13].copy()
    ref_3["st"] = ref_3["st"] - df_ref["st"][8]
    ref_4 = df_ref[12:17].copy()
    ref_4["st"] = ref_4["st"] - df_ref["st"][12]
    return [ref_1, ref_2, ref_3, ref_4]


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
# implement the thrust model accordingly based on the acceleration
# provided in the article
######################################################################
def thrust_model(df_ref, ori_ref):
    volumes = []  # store instantaneous volumes
    masses = []  # store instantaneous masses
    drags = []  # store instantaneous drags
    net_forces = []  # store instantaneous net_forces
    thrusts = []  # store instantaneous thrusts
    dSdt = []  # store instantaneous dSdt

    for row in df_ref.index:
        am = df_ref.at[row, 'am']
        v = df_ref.at[row, 'v']
        r = df_ref.at[row, 're']
        h = df_ref.at[row, 'h']
        d = df_ref.at[row, 'd']
        volumes.append(vol(h, d))
        masses.append(mas(h, d))
        net_forces.append(nf_am(h, d, am))
        drags.append(drg(r, h, d, v))
        thrusts.append(thr_am(h, d, am, r, v))

    df_ref["V"] = volumes
    df_ref["m"] = masses
    df_ref["nfm"] = net_forces
    df_ref["drg"] = drags
    df_ref["tm"] = thrusts

    for row in (list(range(len(df_ref.index) - 1))):
        v1 = df_ref.at[row, 'V']
        v2 = df_ref.at[row + 1, 'V']
        f = df_ref.at[row, 'tm']
        if v1 > v2:
            dSdt.append(-1 * np.sqrt(ori_ref / sea_den * f))
        else:
            dSdt.append(np.sqrt(ori_ref / sea_den * f))

    dSdt.append(0)

    df_ref["dSdt"] = dSdt


######################################################################
# get an improved thrust model by integrating changing orifice area, but
# calculation still based on the acceleration provided in the article
######################################################################
def correct_thrust_model(df_ref):
    orifices = []  # store instantaneous orifice area
    volumes = []  # store instantaneous volumes
    masses = []  # store instantaneous masses
    drags = []  # store instantaneous drags
    net_forces = []  # store instantaneous net_forces
    thrusts = []  # store instantaneous thrusts
    dSdt = []  # store instantaneous dSdt

    for row in df_ref.index:
        am = df_ref.at[row, 'am']
        v = df_ref.at[row, 'v']
        r = df_ref.at[row, 're']
        h = df_ref.at[row, 'h']
        d = df_ref.at[row, 'd']
        orifices.append(ori(d))
        volumes.append(vol(h, d))
        masses.append(mas(h, d))
        net_forces.append(nf_am(h, d, am))
        drags.append(drg(r, h, d, v))
        thrusts.append(thr_am(h, d, am, r, v))

    df_ref["ori"] = orifices
    df_ref["V"] = volumes
    df_ref["m"] = masses
    df_ref["nfm"] = net_forces
    df_ref["drg"] = drags
    df_ref["tm"] = thrusts

    for row in (list(range(len(df_ref.index) - 1))):
        o = df_ref.at[row, 'ori']
        v1 = df_ref.at[row, 'V']
        v2 = df_ref.at[row + 1, 'V']
        f = df_ref.at[row, 'tm']
        if v1 > v2:
            dSdt.append(-1 * np.sqrt(o / sea_den * f))
        else:
            dSdt.append(np.sqrt(o / sea_den * f))

    dSdt.append(0)

    df_ref["dSdt"] = dSdt


######################################################################
# get acceleration based on the published acceleration. used to
# check if the model has been implemented correctly
######################################################################
def get_accel(df_ref, ori_ref):
    new_thrusts = []  # store instantaneous thrust force
    new_net_forces = []  # store instantaneous total force
    accelerations = []  # store instantaneous acceleration

    for row in df_ref.index:
        dSdt = df_ref.at[row, "dSdt"]
        new_thrusts.append(sea_den / ori_ref * np.power(dSdt, 2))

    df_ref["tm"] = new_thrusts

    for row in df_ref.index:
        thr = df_ref.at[row, 'tm']
        drag = df_ref.at[row, 'drg']
        new_net_forces.append(nf(thr, drag))

    df_ref["nfm"] = new_net_forces

    for row in df_ref.index:
        m = df_ref.at[row, 'm']
        f = df_ref.at[row, 'nfm']
        accelerations.append(f / m)

    df_ref["ac"] = accelerations


######################################################################
# get acceleration based on the improved acceleration calculation.
# used to compare to the observed acceleration
######################################################################
def get_correct_accel(df_ref):
    new_thrusts = []  # store instantaneous thrust force
    new_net_forces = []  # store instantaneous total force
    accelerations = []  # store instantaneous acceleration

    for row in df_ref.index:
        o = df_ref.at[row, "ori"]
        dSdt = df_ref.at[row, "dSdt"]
        new_thrusts.append(sea_den / o * np.power(dSdt, 2))

    df_ref["tm"] = new_thrusts

    for row in df_ref.index:
        thr = df_ref.at[row, 'tm']
        drag = df_ref.at[row, 'drg']
        new_net_forces.append(nf(3*thr, drag))

    df_ref["nfm"] = new_net_forces

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
# get net force (g * m / s^2) from given acceleration
# param: bell height(m), bell diameter(m), modeled acceleration(m/s^2)
######################################################################
def nf_am(h_ref, d_ref, am_ref):
    mass = mas(h_ref, d_ref)
    # force = mass * acceleration
    # force: g * m / s^2 = g * (m / s^2)
    net_force = mass * am_ref
    return net_force


######################################################################
# get thrust (g * m / s^2) from given acceleration
# param: bell height(m), bell diameter(m), acceleration(m/s^2),
#        Re, swimming velocity (m/s)
######################################################################
def thr_am(h_ref, d_ref, am_ref, re_ref, u_ref):
    force = nf_am(h_ref, d_ref, am_ref)
    drag = drg(re_ref, h_ref, d_ref, u_ref)
    thrust = force + drag
    if thrust < 0:
        return 0
    else:
        return thrust


######################################################################
# get net force (g * m / s^2) from thrust and drag
# param: thrust (g * m / s^2), drag (g * m / s^2)
######################################################################
def nf(thr_ref, drg_ref):
    # force = thrust - drag
    # force: g * m / s^2 = g * m / s^2 - g * m / s^2
    net_force = thr_ref - drg_ref
    return net_force


main()
