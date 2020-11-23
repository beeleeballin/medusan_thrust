from methods import *


######################################################################
# TASK 3:
# to improve and validate our new model designed to project oblate medusan
# acceleration. the new estimates are compared to the empirical data
# published in the article. similar to the author's validation process, the
# projected results that are statistically indistinguishable from the empirical
# data are considered accurate.
# RESULT:
# projection for 2 oblate accelerations are accurate
######################################################################
def main():
    pd.set_option('display.max_rows', None)

    # import pre-cleaned up csv files, 3 oblates & 11 oblates
    dfs = list()
    for filename in all_csv:
        df = pd.read_csv(filename)
        dfs.append(df)

    # set constant orifice areas for oblate medusae p_gregarium: 2.14cm (Colin & Costello, 2001)
    p_gregarium_ori = ori(2.14 / 100)

    # group the medusa of interest for easy access
    name = ["$S. meleagris$", "$L unguiculata$", "$P. gregarium$", "L. tetraphylla",
            "A. victoria", "A. aurita", "C. capillata", 'more?', 'how?', 'ahh', 'I know why']  # "M. cellularia"
    colors = ['firebrick', 'goldenrod', 'green', 'dodgerblue', 'purple', 'red',
              'orange', 'yellow', 'cyan', 'blue', 'pink']

    fig_lab = ['A', 'B', 'C', 'D']
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
    dfs_count = 0
    for df in dfs[3:5]:
        improved_model(df, p_gregarium_ori)
        if dfs_count == 0:
            ax = ax1
        else:
            ax = ax2
            ax.set_xlabel("Time $(s)$")

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.plot(df.loc[df['st'] <= 1.5, 'st'], df.loc[df['st'] <= 1.5, 'ac'], linewidth=5, label='Model, Improved', color='orange')
        ax.plot(df.loc[df['st'] <= 1.5, 'st'], df.loc[df['st'] <= 1.5, 'am'], label='Model, Published', color='dodgerblue')
        ax.plot(df.loc[df['st'] <= 1.5, 'st'], df.loc[df['st'] <= 1.5, 'ao'], label='Empirical', color='green')
        ax.set_xticks([0, 0.3, 0.6, 0.9, 1.2, 1.5])
        ax.set_ylabel("Acceleration $(m/s^2)$")
        if dfs_count == 1:
            ax.legend(loc='lower right')
        dfs_count += 1

    figtext(0.15, 0.9, 'A', size=25)
    figtext(0.15, 0.45, 'B', size=25)
    plt.tight_layout()
    plt.show()


    # # results for P. gregarium is not ideal. abandon lol
    # for df in dfs[9:10]:
    #     improved_model(df)
    #     print(df)
    #     fig, ax = plt.subplots()
    #     ax.spines['top'].set_visible(False)
    #     ax.spines['right'].set_visible(False)
    #     plt.plot(df["st"], df["ac"], label='Model, Improved', color='orange')
    #     plt.plot(df["st"], df["ao"], label='Empirical', color='green')
    #     plt.legend(loc='lower right')
    #     plt.xlabel("Time $(s)$")
    #     plt.ylabel("Acceleration $(m/s^2)$")
    #     figtext(0.2, 0.9, fig_lab[dfs_count], size=25)
    #     plt.tight_layout()
    #     plt.show()

    dfss = [split_meleagris2(dfs[3]), split_unguiculata2(dfs[4])]

    # max acceleration!
    dfs_count = 0
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
    p_value = []
    for dfs in dfss:
        if dfs_count == 0:
            ax = ax1
            acc_per_cycle = np.empty([15])
        else:
            ax = ax2
            acc_per_cycle = np.empty([12])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        df_count = 0
        for df in dfs:
            acc_per_cycle[df_count * 3] = max(df["am"])
            acc_per_cycle[df_count * 3 + 1] = max(df["ac"])
            acc_per_cycle[df_count * 3 + 2] = max(df["ao"])
            df_count += 1
        max_acc = pd.DataFrame(acc_per_cycle, columns=['accel'])
        acc_type = ['am', 'ac', 'ao']
        for row in max_acc.index:
            max_acc.loc[row, 'type'] = acc_type[row % 3]
        p_value.append(stats.f_oneway(max_acc['accel'][max_acc['type'] == 'am'],
                                      max_acc['accel'][max_acc['type'] == 'ac'],
                                      max_acc['accel'][max_acc['type'] == 'ao'])[1])
        max_acc = max_acc.pivot(columns='type', values='accel')
        max_acc = max_acc.reindex(columns=['am', 'ac', 'ao'])
        max_acc = max_acc.apply(lambda x: pd.Series(x.dropna().values))
        means = max_acc.mean(axis=0)
        errors = max_acc.std(axis=0) / 2
        width = 0.1
        ax.bar(np.arange(3), means, yerr=errors, width=7 * width, label='means', color='dodgerblue')
        if dfs_count == 0:
            ax.bar(np.arange(3) - 2 * width, max_acc.iloc[0], alpha=0.5, width=width, color='green')
            ax.bar(np.arange(3) - 1 * width, max_acc.iloc[1], alpha=0.5, width=width, color='green')
            ax.bar(np.arange(3), max_acc.iloc[2], alpha=0.5, width=width, color='green')
            ax.bar(np.arange(3) + 1 * width, max_acc.iloc[3], alpha=0.5, width=width, color='green')
            ax.bar(np.arange(3) + 2 * width, max_acc.iloc[4], alpha=0.5, width=width, color='green')
        if dfs_count == 1:
            ax.bar(np.arange(3) - 1.5 * width, max_acc.iloc[0], alpha=0.5, width=width, color='green')
            ax.bar(np.arange(3) - 0.5 * width, max_acc.iloc[1], alpha=0.5, width=width, color='green')
            ax.bar(np.arange(3) + 0.5 * width, max_acc.iloc[2], alpha=0.5, width=width, color='green')
            ax.bar(np.arange(3) + 1.5 * width, max_acc.iloc[3], alpha=0.5, width=width, color='green')
        labels = ['Model, Replicated', 'Model, Improved', 'Empirical']
        ax.set_ylabel('Max Acceleration ($m/s^2$)')
        l1 = lines.Line2D([0.41, 0.88], [0.88, 0.88], transform=fig.transFigure, figure=fig, color='black')
        l2 = lines.Line2D([0.41, 0.41], [0.85, 0.88], transform=fig.transFigure, figure=fig, color='black')
        l3 = lines.Line2D([0.88, 0.88], [0.85, 0.88], transform=fig.transFigure, figure=fig, color='black')
        l4 = lines.Line2D([0.41, 0.88], [0.46, 0.46], transform=fig.transFigure, figure=fig, color='black')
        l5 = lines.Line2D([0.41, 0.41], [0.43, 0.46], transform=fig.transFigure, figure=fig, color='black')
        l6 = lines.Line2D([0.88, 0.88], [0.43, 0.46], transform=fig.transFigure, figure=fig, color='black')
        fig.lines.extend([l1, l2, l3, l4, l5, l6])
        figtext(0.37, 0.65, "*", size=20)
        figtext(0.37, 0.2, "*", size=20)
        float_formatter = "{:.2f}".format
        ax.text(0, 5, float_formatter(means[0]), horizontalalignment='center')
        plt.xticks(range(3), labels)
        dfs_count += 1
    figtext(0.15, 0.85, 'C', size=25)
    figtext(0.15, 0.78, "p: %.2E" % p_value[0], size=15)
    figtext(0.15, 0.4, 'D', size=25)
    figtext(0.15, 0.33, "p: %.2E" % p_value[1], size=15)
    plt.show()


    #
    # total_x = np.array([])
    # total_y1 = np.array([])
    # total_y2 = np.array([])
    # count = 0
    # for df in dfs:
    #     label = "cycle #" + str(count + 1)
    #     plt.plot(df["st"], df["ac"], color='magenta', label=label)
    #     plt.plot(df["st"], df["am"], color='purple', label=label)
    #     total_x = np.append(total_x, df["st"].values)
    #     total_y1 = np.append(total_y1, df["ac"].values)
    #     total_y2 = np.append(total_y2, df["am"].values)
    #     count += 1
    # input_regression = [('polynomial', PolynomialFeatures(degree=2)), ('modal', LinearRegression())]
    # pipe1 = Pipeline(input_regression)
    # pipe1.fit(total_x.reshape(-1, 1), total_y1.reshape(-1, 1))
    # poly_pred1 = pipe1.predict(total_x.reshape(-1, 1))
    # sorted_zip1 = sorted(zip(total_x, poly_pred1))
    # x_poly, poly_pred1 = zip(*sorted_zip1)
    # reg_label1 = ('RMSE %f' % np.sqrt(mean_squared_error(x_poly, poly_pred1)))
    # plt.plot(x_poly, poly_pred1, color='magenta', linestyle='--', label=reg_label1)
    # pipe2 = Pipeline(input_regression)
    # pipe2.fit(total_x.reshape(-1, 1), total_y2.reshape(-1, 1))
    # poly_pred2 = pipe2.predict(total_x.reshape(-1, 1))
    # sorted_zip2 = sorted(zip(total_x, poly_pred2))
    # x_poly, poly_pred2 = zip(*sorted_zip2)
    # reg_label2 = ('RMSE %f' % np.sqrt(mean_squared_error(x_poly, poly_pred1)))
    # plt.plot(x_poly, poly_pred2, color='purple', linestyle='--', label=reg_label2)
    # plt.title("p_gregarium acceleration over time")
    # plt.legend()
    # plt.xlabel("time s")
    # plt.ylabel("acceleration ms^-2")
    # plt.tight_layout()
    # plt.show()
    #
    # dfs_count += 1

main()
