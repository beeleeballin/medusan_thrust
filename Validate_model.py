from methods import *


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
    name = ['prolate medusae', 'oblate medusae']

    # set constant orifice areas for medusa
    # s_sp: 0.85cm, p_gregarium: 2.14cm (Colin & Costello, 2001)
    s_sp_ori = ori(0.85 / 100)
    p_gregarium_ori = ori(2.14 / 100)
    # group the two constant orifice areas for easy access
    oris = [s_sp_ori, p_gregarium_ori]

    dfs_count = 0
    for (df, o) in zip(dfs, oris):
        copy_model(df, o)
        plt.plot(df["st"], df["ac"], linewidth=5, label='model estimate', color='orange')
        plt.plot(df["st"], df["am"], label='published estimate', color='dodgerblue')
        plt.plot(df["st"], df["ao"], label='observed acceleration', color='green')
        plt.title("MODEL1 acceleration estimate matches the published %s acceleration estimate" % name[dfs_count])
        plt.legend()
        plt.xlabel("Swimming Time s")
        plt.ylabel("Acceleration m/s^2")
        plt.tight_layout()
        plt.show()
        dfs_count += 1

    dfss = [split_sp(dfs[0]), split_gregarium(dfs[1])]

    # present the maximum acceleration predicted by the published model and MODEL1,
    # along with the observed data of 4 pulsation cycles in a barchart
    dfs_count = 0
    for dfs in dfss:
        acc_per_cycle = np.empty([12])
        df_count = 0
        for df in dfs:
            acc_per_cycle[df_count*3] = max(df["am"])
            acc_per_cycle[df_count*3+1] = max(df["ac"])
            acc_per_cycle[df_count*3+2] = max(df["ao"])
            df_count += 1
        max_acc = pd.DataFrame(acc_per_cycle, columns=['accel'])
        acc_type = ['am', 'ac', 'ao']
        for row in max_acc.index:
            max_acc.loc[row, 'type'] = acc_type[row % 3]
        # print(rp.summary_cont(max_accel['accel'].groupby(max_accel['type'])))
        am_ac_p = stats.f_oneway(max_acc['accel'][max_acc['type'] == 'am'],
                             max_acc['accel'][max_acc['type'] == 'ac'])[1]
        am_ao_p = stats.f_oneway(max_acc['accel'][max_acc['type'] == 'am'],
                                 max_acc['accel'][max_acc['type'] == 'ao'])[1]
        ac_ao_p = stats.f_oneway(max_acc['accel'][max_acc['type'] == 'ac'],
                                 max_acc['accel'][max_acc['type'] == 'ao'])[1]
        max_acc = max_acc.pivot(columns='type', values='accel')
        max_acc = max_acc.reindex(columns=['am', 'ac', 'ao'])
        max_acc = max_acc.apply(lambda x: pd.Series(x.dropna().values))
        means = max_acc.mean(axis=0)
        errors = max_acc.std(axis=0)/2
        width = 0.1
        plt.bar(np.arange(3), means, yerr=errors, width=7 * width, label='means', color='dodgerblue')
        plt.bar(np.arange(3) - 1.5 * width, max_acc.iloc[0], alpha=0.5, width=width, color='green')
        plt.bar(np.arange(3) - 0.5 * width, max_acc.iloc[1], alpha=0.5, width=width, color='green')
        plt.bar(np.arange(3) + 0.5 * width, max_acc.iloc[2], alpha=0.5, width=width, color='green')
        plt.bar(np.arange(3) + 1.5 * width, max_acc.iloc[3], alpha=0.5, width=width, color='green')
        plt.legend()
        labels = ['Published Estimate', 'Model1 Estimate', 'Observed Acceleration']
        plt.ylabel('Max Acceleration m/s^2')
        plt.title("MODEL1 max acceleration estimate matches the published %s acceleration estimate" % name[dfs_count])
        plt.xticks(range(3), labels)
        figtext(0.15, 0.78, "published & modeled:  p = %.2E" % am_ac_p)
        figtext(0.15, 0.74, "published & observed: p = %.2E" % am_ao_p)
        figtext(0.15, 0.7, "modeled & observed: p = %.2E" % ac_ao_p)
        plt.show()
        dfs_count += 1

    # reimport a pre-cleaned up p.gregarium csv as df
    p_gregarium = pd.read_csv(p_gregarium_f)
    # run it through our modified model
    tweaked_model(p_gregarium)

    acc_per_cycle = np.empty([12])
    df_count = 0
    for df in split_gregarium(p_gregarium):
        acc_per_cycle[df_count * 3] = max(df["am"])
        acc_per_cycle[df_count * 3 + 1] = max(df["ac"])
        acc_per_cycle[df_count * 3 + 2] = max(df["ao"])
        df_count += 1
    max_acc = pd.DataFrame(acc_per_cycle, columns=['accel'])
    acc_type = ['am', 'ac', 'ao']
    for row in max_acc.index:
        max_acc.loc[row, 'type'] = acc_type[row % 3]
    am_ac_p = stats.f_oneway(max_acc['accel'][max_acc['type'] == 'am'],
                             max_acc['accel'][max_acc['type'] == 'ac'])[1]
    am_ao_p = stats.f_oneway(max_acc['accel'][max_acc['type'] == 'am'],
                             max_acc['accel'][max_acc['type'] == 'ao'])[1]
    ac_ao_p = stats.f_oneway(max_acc['accel'][max_acc['type'] == 'ac'],
                             max_acc['accel'][max_acc['type'] == 'ao'])[1]
    max_acc = max_acc.pivot(columns='type', values='accel')
    max_acc = max_acc.reindex(columns=['am', 'ac', 'ao'])
    max_acc = max_acc.apply(lambda x: pd.Series(x.dropna().values))
    means = max_acc.mean(axis=0)
    errors = max_acc.std(axis=0) / 2
    width = 0.1
    plt.bar(np.arange(3), means, yerr=errors, width=7 * width, label='means', color='dodgerblue')
    plt.bar(np.arange(3) - 1.5 * width, max_acc.iloc[0], alpha=0.5, width=width, color='green')
    plt.bar(np.arange(3) - 0.5 * width, max_acc.iloc[1], alpha=0.5, width=width, color='green')
    plt.bar(np.arange(3) + 0.5 * width, max_acc.iloc[2], alpha=0.5, width=width, color='green')
    plt.bar(np.arange(3) + 1.5 * width, max_acc.iloc[3], alpha=0.5, width=width, color='green')
    plt.legend()
    labels = ['Published Estimate', 'Model1 Estimate', 'Observed Acceleration']
    plt.ylabel('Max Acceleration m/s^2')
    plt.title("P. gregarium max acceleration estimates by MODEL1 with minor tweaks")
    plt.xticks(range(3), labels)
    figtext(0.15, 0.75, "published & modeled:  p = %.2E" % am_ac_p)
    figtext(0.15, 0.7, "published & observed: p = %.2E" % am_ao_p)
    figtext(0.15, 0.65, "modeled & observed: p = %.2E" % ac_ao_p)
    plt.show()
    dfs_count += 1

    # df_count = 0
    # for df in dfs:
    #     label = " cycle #" + str(df_count + 1)
    #     plt.plot(df["st"], df["am"], color=colors[0], linestyle='--', label=("published"+label))
    #     plt.plot(df["st"], df["ac"], color=colors[1], linestyle='--', label=("corrected"+label))
    #     plt.plot(df["st"], df["ao"], color=colors[2], label=("observed" + label))
    #     df_count += 1
    # plt.title("p_gregarium acceleration estimates by MODEL1 with minor tweaks")
    # plt.legend()
    # plt.xlabel("cycle percentage %")
    # plt.ylabel("acceleration m * s^-2")
    # plt.tight_layout()
    # plt.show()

    # data = [3, 4, np.NaN]
    # print("average is: %f" % np.nanmean(data))


main()
