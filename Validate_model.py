from methods import *


######################################################################
# TASK 1:
# to validate our replicated model. the description of the analysis
# in the published article (Colin & Costello, 2001) is followed religiously,
# and our generated data are compared to the published data to prove our
# implementation works as intended
# RESULT:
# our model provides outputs significantly similar to the published
# results, therefore it is considered a successful program. however,
# as described, while it makes accurate estimates of prolate medusae acceleration
# its projections for oblate medusae acceleration are incorrect.
# an improved model is needed!
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

    fig_lab = ['A', 'B', 'C']

    dfs_count = 0
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
    for (df, o) in zip(dfs, oris):
        copy_model(df, o)
        if dfs_count == 0:
            ax = ax1
        else:
            ax = ax2
            ax.set_xlabel("Time $(s)$")

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.plot(df.loc[df['st'] <= 3, 'st'], df.loc[df['st'] <= 3, 'ac'], linewidth=5, label='Model, Replicated', color='orange')
        ax.plot(df.loc[df['st'] <= 3, 'st'], df.loc[df['st'] <= 3, 'am'], label='Model, Published', color='dodgerblue')
        ax.plot(df.loc[df['st'] <= 3, 'st'], df.loc[df['st'] <= 3, 'ao'], label='Empirical', color='green')
        ax.set_ylabel("Acceleration $(m/s^2)$")
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
        # ax.set_xticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
        if dfs_count == 1:
            ax.legend(loc='lower right')
        dfs_count += 1
    figtext(0.2, 0.9, 'A', size=25)
    figtext(0.2, 0.45, 'B', size=25)
    plt.tight_layout()
    plt.show()

    dfss = [split_sp(dfs[0]), split_gregarium(dfs[1])]

    # present the maximum acceleration predicted by the published model and MODEL1,
    # along with the observed data of 4 pulsation cycles in a barchart
    dfs_count = 0
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
    p_value = []
    for dfs in dfss:
        if dfs_count == 0:
            ax = ax1
        else:
            ax = ax2
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

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
        p_value.append(stats.f_oneway(max_acc['accel'][max_acc['type'] == 'am'],
                                 max_acc['accel'][max_acc['type'] == 'ac'],
                                 max_acc['accel'][max_acc['type'] == 'ao'])[1])
        print(pairwise_tukeyhsd(max_acc['accel'], max_acc['type']))
        max_acc = max_acc.pivot(columns='type', values='accel')
        max_acc = max_acc.reindex(columns=['am', 'ac', 'ao'])
        max_acc = max_acc.apply(lambda x: pd.Series(x.dropna().values))
        means = max_acc.mean(axis=0)
        errors = max_acc.std(axis=0)/2
        width = 0.1
        ax.bar(np.arange(3), means, yerr=errors, width=7 * width, label='means', color='dodgerblue')
        ax.bar(np.arange(3) - 1.5 * width, max_acc.iloc[0], alpha=0.5, width=width, color='green')
        ax.bar(np.arange(3) - 0.5 * width, max_acc.iloc[1], alpha=0.5, width=width, color='green')
        ax.bar(np.arange(3) + 0.5 * width, max_acc.iloc[2], alpha=0.5, width=width, color='green')
        ax.bar(np.arange(3) + 1.5 * width, max_acc.iloc[3], alpha=0.5, width=width, color='green')
        labels = ['Model, Published', 'Model, Replicated', 'Empirical']
        plt.xticks(range(3), labels)
        # plt.xticks(range(3))
        l1 = lines.Line2D([0.15, 0.62], [0.2, 0.2], transform=fig.transFigure, figure=fig, color='black')
        l2 = lines.Line2D([0.15, 0.15], [0.15, 0.2], transform=fig.transFigure, figure=fig, color='black')
        l3 = lines.Line2D([0.62, 0.62], [0.15, 0.2], transform=fig.transFigure, figure=fig, color='black')
        fig.lines.extend([l1, l2, l3])
        figtext(0.63, 0.3, "*", size=20)
        ax.set_ylabel('Max Acceleration $(m/s^2)$')
        dfs_count += 1
    figtext(0.15, 0.85, 'C', size=25)
    figtext(0.25, 0.85, "p: %.2E" % p_value[0], size=15)
    figtext(0.15, 0.4, 'D', size=25)
    figtext(0.25, 0.4, "p: %.2E" % p_value[1], size=15)
    plt.show()


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
    p_value = stats.f_oneway(max_acc['accel'][max_acc['type'] == 'am'],
                             max_acc['accel'][max_acc['type'] == 'ac'],
                             max_acc['accel'][max_acc['type'] == 'ao'])[1]
    print(pairwise_tukeyhsd(max_acc['accel'], max_acc['type']))
    max_acc = max_acc.pivot(columns='type', values='accel')
    max_acc = max_acc.reindex(columns=['am', 'ac', 'ao'])
    max_acc = max_acc.apply(lambda x: pd.Series(x.dropna().values))
    means = max_acc.mean(axis=0)
    errors = max_acc.std(axis=0) / 2
    fig, ax = plt.subplots()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    width = 0.1
    plt.bar(np.arange(3), means, yerr=errors, width=7 * width, label='means', color='dodgerblue')
    plt.bar(np.arange(3) - 1.5 * width, max_acc.iloc[0], alpha=0.5, width=width, color='green')
    plt.bar(np.arange(3) - 0.5 * width, max_acc.iloc[1], alpha=0.5, width=width, color='green')
    plt.bar(np.arange(3) + 0.5 * width, max_acc.iloc[2], alpha=0.5, width=width, color='green')
    plt.bar(np.arange(3) + 1.5 * width, max_acc.iloc[3], alpha=0.5, width=width, color='green')
    labels = ['Model, Published', 'Model, Replicated', 'Empirical']
    plt.ylabel('Max Acceleration $(m/s^2)$')
    plt.xticks(range(3), labels)
    figtext(0.37, 0.3, "*", size=20)
    figtext(0.63, 0.62, "*", size=20)
    figtext(0.15, 0.75, "p: %.2E" % p_value, size=15)
    figtext(0.15, 0.8, fig_lab[dfs_count], size=25)
    plt.show()


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
