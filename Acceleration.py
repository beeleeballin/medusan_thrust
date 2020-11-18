from methods import *


######################################################################
# TASK 3:
# to build MODEL2, an improvement model designed to estimate acceleration
# of oblate medusae, and compare the new estimates to the observed
# values published in the article. the auxillary model and another
# calculation adjustment are implemented. MODEL2 is examined for its
# acceleration over time estimate and maximum thrusts.
# RESULT:
# while the test results for 1 oblate medusae were slightly more favorable
# (it'd be better if we have more data) the other 2 were not.
# a new approach is needed.
######################################################################
def main():
    pd.set_option('display.max_rows', None)

    # import pre-cleaned up csv files, 3 oblates & 11 oblates
    dfs = list()
    for filename in all_csv:
        df = pd.read_csv(filename)
        dfs.append(df)

        # group the medusa of interest for easy access
    name = ["S. meleagris", "L unguiculata", "L. tetraphylla", "A. victoria",
            "P. gregarium", "A. aurita", "C. capillata", 'more?', 'how?', 'ahh', 'I know why']  # "M. cellularia"
    colors = ['firebrick', 'goldenrod', 'green', 'dodgerblue', 'purple', 'red',
              'orange', 'yellow', 'cyan', 'blue', 'pink']

    fig_lab = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'L', 'M']
    dfs_count = 0
    for df in dfs[3:5]:
        improved_model(df)
        print(df)
        fig, ax = plt.subplots()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.plot(df["st"], df["ac"], label='MODEL2 projection', color='orange')
        plt.plot(df["st"], df["ao"], label='Observed', color='green')
        plt.legend(loc='lower right')
        if dfs_count == 1:
            plt.xlabel("Time $(s)$")
        plt.ylabel("Acceleration $(m/s^2)$")
        figtext(0.2, 0.9, fig_lab[dfs_count], size=25)
        plt.tight_layout()
        plt.show()
        dfs_count += 1


    dfss = [split_meleagris2(dfs[3]), split_unguiculata2(dfs[4])]
            # split_tetraphylla(dfs[5]) + split_tetraphylla2(dfs[6]), split_victoria(dfs[7]),
            # split_cellularia(dfs[8]), split_gregarium(dfs[9]), split_aurita(dfs[10]),
            # split_capillata(dfs[11]) + split_capillata2(dfs[12])]

    # for dfs in dfss:
    #     for df in dfs:
    #         for row in df.index:
    #             if df.loc[row, 'ac'] > 1:
    #                 df.drop(row, inplace=True)

    # max acceleration!
    dfs_count = 0
    for dfs in dfss:
        fig, ax = plt.subplots()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if dfs_count == 0:
            acc_per_cycle = np.empty([10])
        else:
            acc_per_cycle = np.empty([8])
        df_count = 0
        for df in dfs:
            acc_per_cycle[df_count * 2] = max(df["ac"])
            acc_per_cycle[df_count * 2 + 1] = max(df["ao"])
            df_count += 1
        max_acc = pd.DataFrame(acc_per_cycle, columns=['accel'])
        acc_type = ['ac', 'ao']
        for row in max_acc.index:
            max_acc.loc[row, 'type'] = acc_type[row % 2]
        ac_ao_p = stats.f_oneway(max_acc['accel'][max_acc['type'] == 'ac'],
                                 max_acc['accel'][max_acc['type'] == 'ao'])[1]
        max_acc = max_acc.pivot(columns='type', values='accel')
        max_acc = max_acc.reindex(columns=['ac', 'ao'])
        max_acc = max_acc.apply(lambda x: pd.Series(x.dropna().values))
        means = max_acc.mean(axis=0)
        errors = max_acc.std(axis=0) / 2
        width = 0.1
        plt.bar(np.arange(2), means, yerr=errors, width=7 * width, label='means', color='dodgerblue')
        if dfs_count == 0:
            plt.bar(np.arange(2) - 2 * width, max_acc.iloc[0], alpha=0.5, width=width, color='green')
            plt.bar(np.arange(2) - 1 * width, max_acc.iloc[1], alpha=0.5, width=width, color='green')
            plt.bar(np.arange(2), max_acc.iloc[2], alpha=0.5, width=width, color='green')
            plt.bar(np.arange(2) + 1 * width, max_acc.iloc[3], alpha=0.5, width=width, color='green')
            plt.bar(np.arange(2) + 2 * width, max_acc.iloc[4], alpha=0.5, width=width, color='green')
        if dfs_count == 1:
            plt.bar(np.arange(2) - 1.5 * width, max_acc.iloc[0], alpha=0.5, width=width, color='green')
            plt.bar(np.arange(2) - 0.5 * width, max_acc.iloc[1], alpha=0.5, width=width, color='green')
            plt.bar(np.arange(2) + 0.5 * width, max_acc.iloc[2], alpha=0.5, width=width, color='green')
            plt.bar(np.arange(2) + 1.5 * width, max_acc.iloc[3], alpha=0.5, width=width, color='green')
        labels = ['MODEL2 Projection', 'Observed']
        plt.ylabel('Max Acceleration ($m/s^2$)')
        # plt.title("MODEL2 max acceleration estimate and observed data for %s" % name[dfs_count])
        plt.xticks(range(2), labels)
        figtext(0.2, 0.9, fig_lab[dfs_count+2], size=25)
        figtext(0.15, 0.78, "p = $%.2E$" % ac_ao_p)
        plt.show()
        dfs_count += 1

    # dfs_count = 0
    # for dfs in dfss:
    #     total_x = np.array([])
    #     total_y = np.array([])
    #     colors = ['goldenrod', 'firebrick', 'forestgreen', 'dodgerblue']
    #     df_count = 0
    #     for df in dfs:
    #         label = "cycle #" + str(df_count + 1)
    #         plt.plot(df["st"], df["tf"], color=colors[df_count], label=label)
    #         total_x = np.append(total_x, df["st"].values)
    #         total_y = np.append(total_y, df["tf"].values)
    #         df_count += 1
    #     input_regression = [('polynomial', PolynomialFeatures(degree=2)), ('modal', LinearRegression())]
    #     pipe = Pipeline(input_regression)
    #     pipe.fit(total_x.reshape(-1, 1), total_y.reshape(-1, 1))
    #     poly_pred = pipe.predict(total_x.reshape(-1, 1))
    #     sorted_zip = sorted(zip(total_x, poly_pred))
    #     x_poly, poly_pred = zip(*sorted_zip)
    #     reg_label = ('RMSE %f' % np.sqrt(mean_squared_error(x_poly, poly_pred)))
    #     plt.plot(x_poly, poly_pred, color='magenta', linestyle='--', label=reg_label)
    #     plt.title("%s thrust over time" % (name[dfs_count]))
    #     plt.legend()
    #     plt.xlabel("time s")
    #     plt.ylabel("thrust g*m*s^-2")
    #     plt.tight_layout()
    #     plt.show()
    #     dfs_count += 1

    # dfs_count = 0
    # for dfs in dfss:
    #     total_x = np.array([])
    #     total_y = np.array([])
    #     colors = ['goldenrod', 'firebrick', 'forestgreen', 'dodgerblue']
    #     df_count = 0
    #     for df in dfs:
    #         label = "cycle #" + str(df_count + 1)
    #         plt.plot(df["st"], df["ac"], color=colors[df_count], label=label)
    #         total_x = np.append(total_x, df["st"].values)
    #         total_y = np.append(total_y, df["ac"].values)
    #         df_count += 1
    #     input_regression = [('polynomial', PolynomialFeatures(degree=2)), ('modal', LinearRegression())]
    #     pipe = Pipeline(input_regression)
    #     pipe.fit(total_x.reshape(-1, 1), total_y.reshape(-1, 1))
    #     poly_pred = pipe.predict(total_x.reshape(-1, 1))
    #     sorted_zip = sorted(zip(total_x, poly_pred))
    #     x_poly, poly_pred = zip(*sorted_zip)
    #     reg_label = ('RMSE %f' % np.sqrt(mean_squared_error(x_poly, poly_pred)))
    #     plt.plot(x_poly, poly_pred, color='magenta', linestyle='--', label=reg_label)
    #     plt.title("%s acceleration over time" % (oblate_name[dfs_count]))
    #     plt.legend()
    #     plt.xlabel("time s")
    #     plt.ylabel("acceleration ms^-2")
    #     plt.tight_layout()
    #     plt.show()
    #     dfs_count += 1
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
