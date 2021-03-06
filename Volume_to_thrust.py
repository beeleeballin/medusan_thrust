from methods import *


######################################################################
# TASK 4:
# to build MODEL3 that gives a more reliable estimates on the maximum
# thrust based on the wake volume, which we suppose is directly correlated to
# 3 times the volume change of an oblate bell. MODEL3 is still inspired by
# MODEL1
######################################################################
def main():
    pd.set_option('display.max_rows', None)

    # import pre-cleaned up csv files, 3 oblates & 11 oblates
    dfs = list()
    for filename in all_csv:
        df = pd.read_csv(filename)
        dfs.append(df)

    # group the medusa of interest for easy access
    name = ["a_digitale", "s_sp", "p_flavicirrata", "s_meleagris", "l_unguiculata",
            "l_tetraphylla", "l_tetraphylla_2", "p_gregarium", "a_aurita",
            "c_capillata", "c_capillata_2", "a_victoria", "m_cellularia"]
    colors = ['firebrick', 'goldenrod', 'green', 'dodgerblue', 'purple', 'red', 'orange',
              'yellow', 'pink',  'cyan', 'blue',  'gray', 'black']

    df_count = 0
    for df in dfs:
        clean_time(df)
        get_basics(df)
        get_thrust(df)
        # print(name[df_count])
        # print(df)
        df_count += 1

    dfss = [split_digitale(dfs[0]), split_sp(dfs[1]), split_flavicirrata(dfs[2]),
            split_meleagris(dfs[3]), split_unguiculata(dfs[4]), split_tetraphylla(dfs[5]),
            split_tetraphylla2(dfs[6]), split_gregarium(dfs[9]), split_aurita(dfs[10]),
            split_capillata(dfs[11]), split_capillata2(dfs[12])] # split_victoria(dfs[7]), split_cellularia(dfs[8])

    for df in dfss[5]:
        for row in df.index:
            if df.at[row, 'V'] > 5e-7:
                df.drop(row, inplace=True)
    for df in dfss[9]:
        for row in df.index:
            if df.at[row, 'V'] > 1e-3:
                df.drop(row, inplace=True)
    for df in dfss[10]:
        for row in df.index:
            if df.at[row, 'V'] > 1e-5:
                df.drop(row, inplace=True)



    # for df in dfss[11]:
    #     for row in df.index:
    #         if df.at[row, 'V'] > 1e-7:
    #             df.drop(row, inplace=True)

    # dfs_count = 0
    # for dfs in dfss:
    #     df_count = 0
    #     for df in dfs:
    #         label = "cycle #" + str(df_count + 1)
    #         plt.plot(df["st"]/df["st"].max(), df["V"], color=colors[df_count], label=label)
    #         df_count += 1
    #     plt.title("%s bell volume over 1 pulsation cycle" % (name[dfs_count]))
    #     plt.xlabel("percentage cycle %")
    #     plt.ylabel("volume m^3")
    #     plt.tight_layout()
    #     plt.show()
    #     dfs_count += 1

    # dfs_count = 0
    # for dfs in dfss:
    #     tf_per_cycle = np.zeros([5])
    #     ds_per_cycle = np.zeros([5])
    #     df_count = 0
    #     for df in dfs:
    #         tf_per_cycle[df_count] = sum(df["tf"])
    #         # print("max tf: %.2Ef and rest tf: %.2Ef" % (max(df["tf"]), sum(df["tf"])-max(df["tf"])))
    #         dv = max(df["V"]) - min(df["V"])
    #         # print("max v: %.2E and min v: %.2Ef" % (max(df["V"]), min(df["V"])))
    #         if dfs_count < 3:
    #             ds_per_cycle[df_count] = ds_dv(dv, False)
    #             # print("name: %s and ds: %.2Ef" % (name[dfs_count], ds_per_cycle[df_count]))
    #         else:
    #             ds_per_cycle[df_count] = np.absolute(ds_dv(dv, True))
    #             # print("name: %s and ds: %.2Ef" % (name[dfs_count], ds_per_cycle[df_count]))
    #         df_count += 1
    #     tf_per_cycle = tf_per_cycle[tf_per_cycle != 0]
    #     ds_per_cycle = ds_per_cycle[ds_per_cycle != 0]
    #     means = np.mean(tf_per_cycle / ds_per_cycle)
    #     errors = np.std(tf_per_cycle / ds_per_cycle, ddof=1) / np.sqrt(np.size(tf_per_cycle))
    #     width = 0.1
    #     plt.bar(dfs_count, means, yerr=errors, width=7 * width, label='means', color='dodgerblue')
    #     cycle_count = 0
    #     while cycle_count < np.size(tf_per_cycle):
    #         plt.bar(dfs_count + cycle_count * width, tf_per_cycle[cycle_count]/ds_per_cycle[cycle_count],
    #                 alpha=0.5, width=width, color='green')
    #         cycle_count += 1
    #     dfs_count += 1
    # plt.legend()
    # # labels = ['Model3 Estimate']
    # plt.ylabel('Max Thrust g*m/s^2')
    # plt.title("MODEL3 max thrust estimate")
    # plt.xticks(range(12))
    # plt.show()


    # dfs_count = 0
    # for dfs in dfss:
    #     total_x = np.array([])
    #     total_y = np.array([])
    #     df_count = 0
    #     for df in dfs:
    #         label = "cycle #" + str(df_count + 1)
    #         plt.plot(df["st"]/df["st"].max(), df["tf"], color=colors[df_count], label=label)
    #         total_x = np.append(total_x, df["st"].values)
    #         total_y = np.append(total_y, df["tf"].values)
    #         df_count += 1
    #     # input_regression = [('polynomial', PolynomialFeatures(degree=2)), ('modal', LinearRegression())]
    #     # pipe = Pipeline(input_regression)
    #     # pipe.fit(total_x.reshape(-1, 1), total_y.reshape(-1, 1))
    #     # poly_pred = pipe.predict(total_x.reshape(-1, 1))
    #     # sorted_zip = sorted(zip(total_x, poly_pred))
    #     # x_poly, poly_pred = zip(*sorted_zip)
    #     # reg_label = ('RMSE %f' % np.sqrt(mean_squared_error(x_poly, poly_pred)))
    #     # plt.plot(x_poly, poly_pred, color='magenta', linestyle='--', label=reg_label)
    #     plt.title("%s thrust over 1 pulsation cycle" % (name[dfs_count]))
    #     plt.xlabel("percentage cycle %")
    #     plt.ylabel("thrust g * m * s^-2")
    #     plt.tight_layout()
    #     plt.show()
    #     dfs_count += 1

    dfs_count = 0
    total_x = np.array([])
    total_y = np.array([])
    fig, ax = plt.subplots()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for dfs in dfss:
        x = np.array([])
        y = np.array([])
        for df in dfs:
            for row in df.index:
                if dfs_count < 3:
                    ej_vol = ds_dv(df.loc[row, 'V'], False, False)
                elif df.loc[row, 'tf'] > 0:
                    ej_vol = ds_dv(df.loc[row, 'V'], True, True)
                else:
                    ej_vol = ds_dv(df.loc[row, 'V'], True, False)
                x = np.append(x, ej_vol)
                y = np.append(y, df.loc[row, 'tf'])
        plt.scatter(x, y, label=name[dfs_count], color=colors[dfs_count])
        total_x = np.append(total_x, x)
        total_y = np.append(total_y, y)
        dfs_count += 1
    total_x2 = sm.add_constant(total_x)
    mod = sm.OLS(total_y, total_x2)
    fii = mod.fit()
    p_value = fii.summary2().tables[1]['P>|t|']
    print(fii.summary())
    # plt.scatter(total_x, total_y)
    r = np.corrcoef(total_x, total_y)
    polymodel = np.poly1d(np.polyfit(total_x, total_y, 1))
    m, b = polymodel
    polyline = np.linspace(min(total_x), max(total_x), 100)
    plt.plot(polyline, polymodel(polyline))
    regr = OLS(total_y, total_x).fit()
    plt.xlabel("Instantaneous Wake Volume ($m^3$)")
    plt.ylabel("Instantaneous Thrust ($mN$)")
    plt.legend(loc='lower right')
    figtext(0.15, 0.86, "$r^2$: %.2f" % np.power(r[0, 1], 2))
    figtext(0.15, 0.82, "y: $%.2E$ x + $%.2f$" % (m, b))
    figtext(0.15, 0.78, "p: $%.3E$" % p_value[1])
    # figtext(0.15, 0.74, "aic: %.2f" % regr.aic)
    plt.tight_layout()
    plt.show()

    tf_model_count = 0
    for i in range(3):
        dfs_count = 0
        total_x = np.array([])
        total_y = np.array([])
        fig, ax = plt.subplots()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for dfs in dfss:
            x = np.array([])
            y = np.array([])
            for df in dfs:
                if tf_model_count == 0:
                    bel_vol_change = df["V"].max() - df["V"].min()
                    if dfs_count < 3:
                        ej_vol = ds_dv(bel_vol_change, False, False)
                    else:
                        ej_vol = ds_dv(bel_vol_change, True, True)
                    x = np.append(x, ej_vol)

                    # quadratic mean of inst thrust
                    tf_sq = []
                    for row in df.index:
                        if df.loc[row, 'tf'] > 0:
                            tf_sq.append(np.power(df.loc[row, 'tf'], 2))
                    tf_a = np.sqrt(sum(tf_sq) / len(df.index))
                    y = np.append(y, tf_a)

                if tf_model_count == 1:
                    bel_vol_change = df["V"].min() - df["V"].max()
                    if dfs_count < 3:
                        ej_vol = ds_dv(bel_vol_change, False, False)
                    else:
                        ej_vol = ds_dv(bel_vol_change, True, False)
                    x = np.append(x, ej_vol)

                    # quadratic mean of inst thrust
                    tf_sq = []
                    for row in df.index:
                        if df.loc[row, 'tf'] <= 0:
                            tf_sq.append(np.power(df.loc[row, 'tf'], 2))
                    tf_a = np.sqrt(sum(tf_sq) / len(df.index))
                    y = np.append(y, (-1) * tf_a)

                if tf_model_count == 2:
                    bel_vol_change = df["V"].max() - df["V"].min()
                    x = np.append(x, bel_vol_change)

                    # quadratic mean of inst thrust
                    tf_sq = []
                    pos_row = 0
                    for row in df.index:
                        if df.loc[row, 'tf'] > 0:
                            tf_sq.append(np.power(df.loc[row, 'tf'], 2))
                            pos_row += 1
                    tf_a_p = np.sqrt(sum(tf_sq) / pos_row)
                    tf_sq = []
                    for row in df.index:
                        if df.loc[row, 'tf'] <= 0:
                            tf_sq.append(np.power(df.loc[row, 'tf'], 2))
                    tf_a_n = np.sqrt(sum(tf_sq) / (len(df.index)-pos_row))
                    pos_ratio = pos_row/len(df.index)
                    y = np.append(y, tf_a_p * pos_ratio - tf_a_n * (1-pos_ratio))

            plt.scatter(x, y, label=name[dfs_count], color=colors[dfs_count])
            total_x = np.append(total_x, x)
            total_y = np.append(total_y, y)
            dfs_count += 1
        total_x2 = sm.add_constant(total_x)
        mod = sm.OLS(total_y, total_x2)
        fii = mod.fit()
        p_value = fii.summary2().tables[1]['P>|t|']
        print(fii.summary())
        # plt.scatter(total_x, total_y)
        r = np.corrcoef(total_x, total_y)
        polymodel = np.poly1d(np.polyfit(total_x, total_y, 1))
        m, b = polymodel
        polyline = np.linspace(min(total_x), max(total_x), 100)
        plt.plot(polyline, polymodel(polyline))
        regr = OLS(total_y, total_x).fit()
        if tf_model_count == 0:
            plt.xlabel("Ejected Wake Volume ($m^3$)")
            plt.ylabel("Average Thrust During Propulsion Strokes ($mN$)")
        if tf_model_count == 1:
            plt.xlabel("Inwards Fluid Volume ($m^3$)")
            plt.ylabel("Average Thrust During Recovery Strokes ($mN$)")
        if tf_model_count == 2:
            plt.xlabel("Bell Volume ($m^3$)")
            plt.ylabel("Average Thrust ($mN$)")
        plt.legend(loc='lower right')
        figtext(0.15, 0.86, "$r^2$: %.2f" % np.power(r[0, 1], 2))
        figtext(0.15, 0.82, "y: $%.2E$ x + $%.2f$" % (m, b))
        figtext(0.15, 0.78, "p: $%.3E$" % p_value[1])
        # figtext(0.15, 0.74, "aic: %.2f" % regr.aic)
        plt.tight_layout()
        plt.show()
        tf_model_count += 1


    dfs_count = 0
    total_x = np.array([])
    total_y = np.array([])
    fig, ax = plt.subplots()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for dfs in dfss:
        x = np.array([])
        y = np.array([])
        for df in dfs:
            bel_vol_change = df["V"].max() - df["V"].min()
            if dfs_count < 3:
                ej_vol = ds_dv(bel_vol_change, False)
            else:
                ej_vol = ds_dv(bel_vol_change, True)
            x = np.append(x, ej_vol)

            # quadratic mean of inst thrust
            df['tf_sq'] = np.power(df['tf'], 2)
            tf_a = np.sqrt(sum(df['tf_sq'])/len(df.index))

            # impulse
            df['dt'] = df["st"].diff(1)
            rei_sum = df['tf'] * df.loc[:, "dt"].mean()
            imp = sum(rei_sum)

            # avg_tf = imp / time_change

            y = np.append(y, imp)
        plt.scatter(x, y, label=name[dfs_count], color=colors[dfs_count])
        total_x = np.append(total_x, x)
        total_y = np.append(total_y, y)
        dfs_count += 1
    total_x2 = sm.add_constant(total_x)
    mod = sm.OLS(total_y, total_x2)
    fii = mod.fit()
    p_value = fii.summary2().tables[1]['P>|t|']
    print(fii.summary())
    # plt.scatter(total_x, total_y)
    r = np.corrcoef(total_x, total_y)
    polymodel = np.poly1d(np.polyfit(total_x, total_y, 1))
    m, b = polymodel
    polyline = np.linspace(min(total_x), max(total_x), 100)
    plt.plot(polyline, polymodel(polyline))
    regr = OLS(total_y, total_x).fit()
    plt.xlabel("Wake Volume ($m^3$)")
    plt.ylabel("Impulse ($mN*s$)")
    plt.legend(loc='lower right')
    figtext(0.15, 0.86, "$r^2$: %.2f" % np.power(r[0, 1], 2))
    figtext(0.15, 0.82, "y: $%.2E$ x + $%.2f$" % (m, b))
    figtext(0.15, 0.78, "p: $%.3E$" % p_value[1])
    figtext(0.15, 0.74, "aic: %.2f" % regr.aic)
    plt.tight_layout()
    plt.show()


main()
