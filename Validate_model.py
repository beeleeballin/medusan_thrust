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


def main():
    ######################################################################
    # task 2: validate the model via implementation
    ######################################################################
    # import pre-cleaned up csv files, 1 prolate 1 oblates
    s_sp = pd.read_csv(s_sp_f)
    p_gregarium = pd.read_csv(p_gregarium_f)
    # group the two medusae of interest for easy access
    dfs = [s_sp, p_gregarium]
    name = ['s_sp', 'p_gregarium']

    # set constant orifice areas for medusa
    # s_sp: 0.85cm, p_gregarium: 2.14cm (Colin & Costello, 2001)
    s_sp_ori = ori(0.85 / 100)
    p_gregarium_ori = ori(2.14 / 100)
    # group the two constant orifice areas for easy access
    oris = [s_sp_ori, p_gregarium_ori]

    copy_model(dfs, oris, name)
    dfss = [split_sp(dfs[0]), split_gregarium(dfs[1])]

    colors = ['goldenrod', 'firebrick', 'forestgreen', 'dodgerblue']
    dfs_count = 0
    for dfs in dfss:
        total_x = np.array([])
        total_y = np.array([])
        df_count = 0
        for df in dfs:
            label = "cycle #" + str(df_count + 1)
            plt.plot(df["st"], df["V"], color=colors[df_count], label=label)
            total_x = np.append(total_x, df["st"].values)
            total_y = np.append(total_y, df["V"].values)
            df_count += 1
        polymodel1 = np.poly1d(np.polyfit(total_x, total_y, 2))
        polyline1 = np.linspace(min(total_x), max(total_x), 100)
        plt.plot(polyline1, polymodel1(polyline1), color='magenta', linestyle='--', label='p(2) regression')
        plt.title("modeled %s bell volume per cycle" % (name[dfs_count]))
        plt.legend()
        plt.xlabel("time s")
        plt.ylabel("volume m^3")
        plt.tight_layout()
        plt.show()

        dfs_count += 1

    dfs_count = 0
    for dfs in dfss:
        df_count = 0
        for df in dfs:
            label = " #" + str(df_count + 1)
            plt.plot(df["st"], df["am"], color='firebrick', label=("published" + label))
            plt.plot(df["st"], df["ac"], color='forestgreen', label=("modeled" + label))
            plt.plot(df["st"], df["ao"], color='goldenrod', label=("observed" + label))
            df_count += 1
        plt.title("modeled %s per cycle acceleration by published model matches published data" % (name[dfs_count]))
        plt.legend()
        plt.xlabel("time s")
        plt.ylabel("acceleration m * s^-2")
        plt.tight_layout()
        plt.show()

        dfs_count += 1

    # dfs_count = 0
    # for dfs in dfss:
    #     acc_per_cycle = np.empty([4, 3])
    #     df_count = 0
    #     for df in dfs:
    #         acc_per_cycle[df_count][0] = sum(df["am"])/len(df.index)
    #         acc_per_cycle[df_count][1] = sum(df["ac"])/len(df.index)
    #         acc_per_cycle[df_count][2] = sum(df["ao"])/len(df.index)
    #         df_count += 1
    #     labels = ['published', 'corrected', 'observed']
    #     plt.xlabel('3 acceleration outputs')
    #     plt.ylabel('acceleration per cycle m*s^-2')
    #     plt.title('modeled %s average acceleration by published model' % name[dfs_count])
    #     plt.xticks(range(3), labels)
    #     width = 0.2
    #     plt.bar(np.arange(3), acc_per_cycle[0], width=width, label='cycle #1')
    #     plt.bar(np.arange(3) + width, acc_per_cycle[1], width=width, label='cycle #2')
    #     plt.bar(np.arange(3) + 2*width, acc_per_cycle[2], width=width, label='cycle #3')
    #     plt.bar(np.arange(3) + 3*width, acc_per_cycle[3], width=width, label='cycle #4')
    #     plt.legend()
    #     plt.show()
    #     dfs_count += 1

    # acc = np.zeros([3])
    # acc[0] = sum(p_gregarium["am"]) / len(p_gregarium.index)
    # acc[1] = sum(p_gregarium["ac"]) / len(p_gregarium.index)
    # acc[2] = sum(p_gregarium["ao"]) / len(p_gregarium.index)
    # labels = ['published', 'corrected', 'observed']
    # plt.xlabel('3 acceleration outputs')
    # plt.ylabel('acceleration per cycle m*s^-2')
    # plt.title('p_gregarium average acceleration by published model')
    # plt.xticks(range(3), labels)
    # width = 0.5
    # plt.bar(np.arange(3), acc, width=width, label='average acceleration')
    # plt.show()

    dfs_count = 0
    for dfs in dfss:
        acc_per_cycle = np.empty([4, 3])
        df_count = 0
        for df in dfs:
            acc_per_cycle[df_count][0] = max(df["am"])
            acc_per_cycle[df_count][1] = max(df["ac"])
            acc_per_cycle[df_count][2] = max(df["ao"])
            df_count += 1
        means = [np.mean(acc_per_cycle[:, 0]), np.mean(acc_per_cycle[:, 1]), np.mean(acc_per_cycle[:, 2])]
        errors = [np.std(acc_per_cycle[:, 0])/2, np.std(acc_per_cycle[:, 1])/2, np.std(acc_per_cycle[:, 2])/2]
        labels = ['published', 'corrected', 'observed']
        plt.xlabel('3 acceleration outputs')
        plt.ylabel('max acceleration in every cycle m*s^-2')
        plt.title('%s maximum acceleration per cycle by improved model' % name[dfs_count])
        plt.xticks(range(3), labels)
        width = 0.15
        plt.bar(np.arange(3), acc_per_cycle[0], width=width, label='cycle #1')
        plt.bar(np.arange(3) + width, acc_per_cycle[1], width=width, label='cycle #2')
        plt.bar(np.arange(3) + 2 * width, acc_per_cycle[2], width=width, label='cycle #3')
        plt.bar(np.arange(3) + 3 * width, acc_per_cycle[3], width=width, label='cycle #4')
        plt.bar(np.arange(3) + 4 * width, means, yerr=errors, width=width, label='average')
        plt.legend()
        plt.show()

        dfs_count += 1


    ######################################################################
    # task 3: validate our improved model by tweaking the tested
    # implementation of the published model
    ######################################################################
    # reimport a pre-cleaned up p.gregarium csv as df
    p_gregarium = pd.read_csv(p_gregarium_f)
    # run it through our modified model
    improved_model(p_gregarium)
    dfs = split_gregarium(p_gregarium)

    df_count = 0
    for df in dfs:
        label = " cycle #" + str(df_count + 1)
        plt.plot(df["st"], df["ao"], color='firebrick', label=("observed"+label))
        plt.plot(df["st"], df["am"], color='firebrick', linestyle='--', label=("published"+label))
        plt.plot(df["st"], df["ac"], color='forestgreen', linestyle='--', label=("corrected"+label))
        df_count += 1
    plt.title("modeled p_gregarium acceleration per cycle by improved model")
    plt.legend()
    plt.xlabel("time s")
    plt.ylabel("acceleration m * s^-2")
    plt.tight_layout()
    plt.show()

    # acc_per_cycle = np.empty([4, 3])
    # acc = np.zeros([3])
    # df_count = 0
    # for df in dfs:
    #     acc_per_cycle[df_count][0] = sum(df["am"])/len(df.index)
    #     acc_per_cycle[df_count][1] = sum(df["ac"])/len(df.index)
    #     acc_per_cycle[df_count][2] = sum(df["ao"])/len(df.index)
    #     acc[0] += sum(df["am"]) / len(df.index)
    #     acc[1] += sum(df["ac"]) / len(df.index)
    #     acc[2] += sum(df["ao"]) / len(df.index)
    #     df_count += 1
    # labels = ['published', 'corrected', 'observed']
    # plt.xlabel('3 acceleration outputs')
    # plt.ylabel('acceleration per cycle m*s^-2')
    # plt.title('p_gregarium average acceleration by improved model')
    # plt.xticks(range(3), labels)
    # width = 0.15
    # plt.bar(np.arange(3), acc_per_cycle[0], width=width, label='cycle #1')
    # plt.bar(np.arange(3) + width, acc_per_cycle[1], width=width, label='cycle #2')
    # plt.bar(np.arange(3) + 2*width, acc_per_cycle[2], width=width, label='cycle #3')
    # plt.bar(np.arange(3) + 3*width, acc_per_cycle[3], width=width, label='cycle #4')
    # plt.bar(np.arange(3) + 4*width, acc/4, width=width, label='average')
    # plt.legend()
    # plt.show()

    # acc = np.zeros([3])
    # acc[0] = sum(p_gregarium["am"]) / len(p_gregarium.index)
    # acc[1] = sum(p_gregarium["ac"]) / len(p_gregarium.index)
    # acc[2] = sum(p_gregarium["ao"]) / len(p_gregarium.index)
    # labels = ['published', 'corrected', 'observed']
    # plt.xlabel('3 acceleration outputs')
    # plt.ylabel('acceleration per cycle m*s^-2')
    # plt.title('p_gregarium average acceleration by improved model')
    # plt.xticks(range(3), labels)
    # width = 0.5
    # plt.bar(np.arange(3), acc, width=width, label='average acceleration')
    # plt.show()

    acc_per_cycle = np.empty([4, 3])
    df_count = 0
    for df in dfs:
        acc_per_cycle[df_count][0] = max(df["am"])
        acc_per_cycle[df_count][1] = max(df["ac"])
        acc_per_cycle[df_count][2] = max(df["ao"])
        df_count += 1
    means = [np.mean(acc_per_cycle[:, 0]), np.mean(acc_per_cycle[:, 1]), np.mean(acc_per_cycle[:, 2])]
    errors = [np.std(acc_per_cycle[:, 0]) / 2, np.std(acc_per_cycle[:, 1]) / 2, np.std(acc_per_cycle[:, 2]) / 2]
    labels = ['published', 'corrected', 'observed']
    plt.xlabel('3 acceleration outputs')
    plt.ylabel('max acceleration in every cycle m*s^-2')
    plt.title('p_gregarium maximum acceleration per cycle by improved model')
    plt.xticks(range(3), labels)
    width = 0.15
    plt.bar(np.arange(3), acc_per_cycle[0], width=width, label='cycle #1')
    plt.bar(np.arange(3) + width, acc_per_cycle[1], width=width, label='cycle #2')
    plt.bar(np.arange(3) + 2 * width, acc_per_cycle[2], width=width, label='cycle #3')
    plt.bar(np.arange(3) + 3 * width, acc_per_cycle[3], width=width, label='cycle #4')
    plt.bar(np.arange(3) + 4 * width, means, yerr=errors, width=width, label='average')
    plt.legend()
    plt.show()


######################################################################
# run my implementation of their model
# param: dfs, constant orifice, and species names
######################################################################
def copy_model(dfs_ref, oris_ref, name_ref):
    count = 0
    for (df, o) in zip(dfs_ref, oris_ref):
        clean_time(df)
        get_basics(df)
        get_thrust(df, o)
        get_accel(df)
        # the last row of dSdt are zeroed because it requires the change in time
        # and because it would effect the output we ignore that last row entirely
        df.drop(df.tail(1).index, inplace=True)

        plt.plot(df["st"], df["ac"], label='modeled acceleration')
        plt.plot(df["st"], df["am"], label='published acceleration')
        plt.plot(df["st"], df["ao"], label='observed acceleration')
        plt.title("modeled %s acceleration over 4 cycles by published model matches published data" % name_ref[count])
        plt.legend()
        plt.xlabel("time s")
        plt.ylabel("acceleration m/s^2")
        plt.tight_layout()
        plt.show()
        count += 1


######################################################################
# run my implementation of their model
# param: dfs
######################################################################
def improved_model(df_ref):

    clean_time(df_ref)
    get_basics(df_ref)
    get_thrust(df_ref)
    get_accel(df_ref, True)
    df_ref.drop(df_ref.tail(1).index, inplace=True)

    plt.plot(df_ref["st"], df_ref["ac"], label='modeled acceleration')
    plt.plot(df_ref["st"], df_ref["am"], label='published acceleration')
    plt.plot(df_ref["st"], df_ref["ao"], label='observed acceleration')
    plt.title("modeled p_gregarium acceleration over 4 cycles by improved model is closes gap with observed model")
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
# add instantaneous heights, diameters, and update velocities and
# accelerations to the corrected units
# param: list of medusae dataframes
######################################################################
def get_basics(df_ref):
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
        d_h = bell_dim(df_ref.at[row, 're'], u, df_ref.at[row, 'f'])  # re: m^2/s / m^2/s, fineness: m/m
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
# implement the original and improved thrust model based on the modeled
# acceleration provided in the article. improved model uses instantaneous
# orifice area
######################################################################
def get_thrust(df_ref, ori_ref=None):
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
        if ori_ref is None:
            orifices.append(ori(d))
        else:
            orifices.append(ori_ref)
        volumes.append(bell_vol(h, d))
        masses.append(bell_mas(h, d))
        net_forces.append(nf_am(h, d, am))
        drags.append(bell_drag(r, h, d, v))
        thrusts.append(tf_am(h, d, am, r, v))

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
            dSdt.append(-1 * np.sqrt(o / sea_den * f))
        else:
            dSdt.append(np.sqrt(o / sea_den * f))

    dSdt.append(0)

    df_ref["dSdt"] = dSdt


######################################################################
# get acceleration based on the modeled acceleration estimate.
# used to check if the model has been implemented correctly
# and compare to the observed acceleration
######################################################################
def get_accel(df_ref, improved=False):
    new_thrusts = []  # store instantaneous thrust force
    new_net_forces = []  # store instantaneous total force
    accelerations = []  # store instantaneous acceleration

    for row in df_ref.index:
        o = df_ref.at[row, "ori"]
        dSdt = df_ref.at[row, "dSdt"]
        new_thrusts.append(sea_den / o * np.power(dSdt, 2))

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
# get net force (g * m / s^2) from given acceleration
# param: bell height(m), bell diameter(m), modeled acceleration(m/s^2)
######################################################################
def nf_am(h_ref, d_ref, am_ref):
    mass = bell_mas(h_ref, d_ref)
    # force = mass * acceleration
    # force: g * m / s^2 = g * (m / s^2)
    net_force = mass * am_ref
    return net_force


######################################################################
# get thrust (g * m / s^2) from given acceleration
# param: bell height(m), bell diameter(m), acceleration(m/s^2),
#        Re, swimming velocity (m/s)
######################################################################
def tf_am(h_ref, d_ref, am_ref, re_ref, u_ref):
    force = nf_am(h_ref, d_ref, am_ref)
    drag = bell_drag(re_ref, h_ref, d_ref, u_ref)
    thrust = force + drag
    if thrust < 0:
        return 0
    else:
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


main()
