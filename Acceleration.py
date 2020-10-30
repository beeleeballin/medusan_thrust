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
    # import pre-cleaned up csv files, 3 oblates
    # a_digitale = pd.read_csv(a_digitale_f)
    # s_sp = pd.read_csv(s_sp_f)
    # p_flavicirrata = pd.read_csv(p_flavicirrata_f)
    # import pre-cleaned up csv files, 3 oblates
    a_victoria = pd.read_csv(a_victoria_f)
    m_cellularia = pd.read_csv(m_cellularia_f)
    p_gregarium = pd.read_csv(p_gregarium_f)
    # group the medusa of interest for easy access
    dfs = [a_victoria, m_cellularia, p_gregarium]  # a_digitale, s_sp, p_flavicirrata,
    name = ["A. victoria", "M. cellularia", "P. gregarium"]  # "a_digitale", "s_sp", "p_flavicirrata",

    # oris= [(ori(0.83 / 100)), (ori(0.85 / 100)), (ori(0.56 / 100)),
    #        (ori(5 / 100)), (ori(6.5 / 100)), (ori(2.14 / 100))]

    for df in dfs:
        improved_model(df)

    dfss = [split_victoria(dfs[0]), split_cellularia(dfs[1]), split_gregarium(dfs[2])]

    for dfs in dfss:
        for df in dfs:
            for row in df.index:
                if df.loc[row, 'ac'] > 1:
                    df.drop(row, inplace=True)

    dfs_count = 0
    for dfs in dfss:
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
        plt.bar(np.arange(2) - 1.5 * width, max_acc.iloc[0], alpha=0.5, width=width, color='green')
        plt.bar(np.arange(2) - 0.5 * width, max_acc.iloc[1], alpha=0.5, width=width, color='green')
        plt.bar(np.arange(2) + 0.5 * width, max_acc.iloc[2], alpha=0.5, width=width, color='green')
        plt.bar(np.arange(2) + 1.5 * width, max_acc.iloc[3], alpha=0.5, width=width, color='green')
        plt.legend()
        labels = ['MODEL2 Estimate', 'Observed Acceleration']
        plt.ylabel('Max Acceleration m/s^2')
        plt.title("MODEL2 max acceleration estimate and observed data for %s" % name[dfs_count])
        plt.xticks(range(2), labels)
        figtext(0.15, 0.78, "modeled & observed: p = %.2E" % ac_ao_p)
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
