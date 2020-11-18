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
    fig_lab = ['A', 'B', 'C']

    for (df, o) in zip(dfs, oris):
        copy_model(df, o)
        fig, ax = plt.subplots()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.plot(df.loc[df['st'] < 3, 'st'], df.loc[df['st'] < 3, 'ac'], linewidth=5, label='model estimate', color='orange')
        plt.plot(df.loc[df['st'] < 3, 'st'], df.loc[df['st'] < 3, 'am'], label='published estimate', color='dodgerblue')
        plt.plot(df.loc[df['st'] < 3, 'st'], df.loc[df['st'] < 3, 'ao'], label='observed acceleration', color='green')
        plt.legend(loc='lower right')
        if dfs_count != 0:
            plt.xlabel("Time $(s)$")
        plt.ylabel("Acceleration $(m/s^2)$")
        ax.set_xticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
        figtext(0.2, 0.9, fig_lab[dfs_count], size=25)
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
        p_value = stats.f_oneway(max_acc['accel'][max_acc['type'] == 'am'],
                                 max_acc['accel'][max_acc['type'] == 'ac'],
                                 max_acc['accel'][max_acc['type'] == 'ao'])[1]
        print(pairwise_tukeyhsd(max_acc['accel'], max_acc['type']))
        max_acc = max_acc.pivot(columns='type', values='accel')
        max_acc = max_acc.reindex(columns=['am', 'ac', 'ao'])
        max_acc = max_acc.apply(lambda x: pd.Series(x.dropna().values))
        means = max_acc.mean(axis=0)
        errors = max_acc.std(axis=0)/2
        width = 0.1
        fig, ax = plt.subplots()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.bar(np.arange(3), means, yerr=errors, width=7 * width, label='means', color='dodgerblue')
        plt.bar(np.arange(3) - 1.5 * width, max_acc.iloc[0], alpha=0.5, width=width, color='green')
        plt.bar(np.arange(3) - 0.5 * width, max_acc.iloc[1], alpha=0.5, width=width, color='green')
        plt.bar(np.arange(3) + 0.5 * width, max_acc.iloc[2], alpha=0.5, width=width, color='green')
        plt.bar(np.arange(3) + 1.5 * width, max_acc.iloc[3], alpha=0.5, width=width, color='green')
        labels = ['', '', '']
        plt.ylabel('Max Acceleration $(m/s^2)$')
        plt.xticks(range(3), labels)
        # plt.xticks(range(3))
        if dfs_count > 0:
            l1 = lines.Line2D([0.15, 0.62], [0.3, 0.3], transform=fig.transFigure, figure=fig, color='black')
            l2 = lines.Line2D([0.15, 0.15], [0.23, 0.3], transform=fig.transFigure, figure=fig, color='black')
            l3 = lines.Line2D([0.62, 0.62], [0.23, 0.3], transform=fig.transFigure, figure=fig, color='black')
            fig.lines.extend([l1, l2, l3])
            figtext(0.63, 0.4, "*", size=20)
        figtext(0.15, 0.75, "p: %.2E" % p_value, size=15)
        figtext(0.15, 0.8, fig_lab[dfs_count], size=25)
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
    labels = ['Published Estimate', 'MODEL1 Estimate', 'Observed Acceleration']
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
