import pandas as pd
import numpy as np
import re
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from methods import *

# global variables

f_path = "/Users/beelee/PycharmProjects/OblateThrust/csv/"
s_sp_f = f_path + "s_sp_data.csv"
p_gregarium_f = f_path + "p_gregarium_data.csv"


######################################################################
# TASK 1:
# to validate the model implementation. the description of the analysis
# in the published article (Colin & Costello, 2001) is followed religiously,
# and results are compared to the published data to prove the model
# works as intended. the program below runs MODEL1.
# RESULT:
# MODEL1 provides outputs significantly similar to the published
# results, therefore MODEL1 is deemed successfully programmed. however,
# while MODEL1 makes accurate estimates of prolate medusae acceleration
# (and thrust) as intended, the estimates on oblate medusae acceleration
# remain incorrect. a new model is needed.
######################################################################
def main():

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

    dfs_count = 0
    for (df, o) in zip(dfs, oris):
        copy_model(df, o)
        plt.plot(df["st"], df["ac"], label='modeled acceleration')
        plt.plot(df["st"], df["am"], label='published acceleration')
        plt.plot(df["st"], df["ao"], label='observed acceleration')
        plt.title("MODEL1 acceleration estimates match published %s modeled acceleration data" % name[dfs_count])
        plt.legend()
        plt.xlabel("time s")
        plt.ylabel("acceleration m/s^2")
        plt.tight_layout()
        plt.show()
        dfs_count += 1

    dfss = [split_sp(dfs[0]), split_gregarium(dfs[1])]

    colors = ['goldenrod', 'firebrick', 'forestgreen', 'dodgerblue']

    dfs_count = 0
    for dfs in dfss:
        df_count = 0
        for df in dfs:
            label = " #" + str(df_count + 1)
            plt.plot(df["st"], df["am"], color=colors[0], linestyle='--', label=("published" + label))
            plt.plot(df["st"], df["ac"], color=colors[1], linestyle='--', label=("modeled" + label))
            plt.plot(df["st"], df["ao"], color=colors[2], label=("observed" + label))
            df_count += 1
        plt.title("MODEL1 acceleration estimates match published %s modeled acceleration data" % name[dfs_count])
        plt.legend()
        plt.xlabel("cycle percentage %")
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
        labels = ['published', 'modeled', 'observed']
        plt.xlabel('3 acceleration data')
        plt.ylabel('max acceleration per cycle m*s^-2')
        plt.title("MODEL1 max acceleration estimates match published %s modeled acceleration data" % name[dfs_count])
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

    # reimport a pre-cleaned up p.gregarium csv as df
    p_gregarium = pd.read_csv(p_gregarium_f)
    # run it through our modified model
    tweaked_model(p_gregarium)

    plt.plot(p_gregarium["st"], p_gregarium["ac"], label='modeled acceleration')
    plt.plot(p_gregarium["st"], p_gregarium["am"], label='published acceleration')
    plt.plot(p_gregarium["st"], p_gregarium["ao"], label='observed acceleration')
    plt.title("p_gregarium acceleration estimates by MODEL1 with minor tweaks")
    plt.legend()
    plt.xlabel("time s")
    plt.ylabel("acceleration m/s^2")
    plt.tight_layout()
    plt.show()

    dfs = split_gregarium(p_gregarium)

    df_count = 0
    for df in dfs:
        label = " cycle #" + str(df_count + 1)
        plt.plot(df["st"], df["am"], color=colors[0], linestyle='--', label=("published"+label))
        plt.plot(df["st"], df["ac"], color=colors[1], linestyle='--', label=("corrected"+label))
        plt.plot(df["st"], df["ao"], color=colors[2], label=("observed" + label))
        df_count += 1
    plt.title("p_gregarium acceleration estimates by MODEL1 with minor tweaks")
    plt.legend()
    plt.xlabel("cycle percentage %")
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
    labels = ['published', 'improved', 'observed']
    plt.xlabel('3 acceleration data')
    plt.ylabel('max acceleration per cycle m * s^-2')
    plt.title("p_gregarium max acceleration estimates by MODEL1 with minor tweaks")
    plt.xticks(range(3), labels)
    width = 0.15
    plt.bar(np.arange(3), acc_per_cycle[0], width=width, label='cycle #1')
    plt.bar(np.arange(3) + width, acc_per_cycle[1], width=width, label='cycle #2')
    plt.bar(np.arange(3) + 2 * width, acc_per_cycle[2], width=width, label='cycle #3')
    plt.bar(np.arange(3) + 3 * width, acc_per_cycle[3], width=width, label='cycle #4')
    plt.bar(np.arange(3) + 4 * width, means, yerr=errors, width=width, label='average')
    plt.legend()
    plt.show()


main()
