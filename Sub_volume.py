from methods import *


######################################################################
# TASK 2:
# to make adjustments to MODEL1 and validate the proposed improvement,
# an extra parameter is required and yet it is not accessible.
# the change in the bell's subumbrellar volume per pulsation is crucial
# for building MODEL2. an auxillary model is constructed to make sound
# estimates of that parameter based the change of bell volume per pulsation
# RESULT:
######################################################################
def main():
    pd.set_option('display.max_rows', None)

    # import pre-cleaned up csv files, 3 oblates & 11 oblates
    dfs = list()
    for filename in all_csv:
        df = pd.read_csv(filename)
        dfs.append(df)

        # group the medusa of interest for easy access
    name = ["A. victoria", "P. gregarium", "A. aurita", "C. capillata",
            "A. digitale", "Sarsia sp.", "P. flavicirrata"]
            #"M. cellularia", "S. meleagris", "L unguiculata", "L. tetraphylla",
    colors = ['firebrick', 'goldenrod', 'green', 'dodgerblue', 'purple', 'magenta',
              'cyan', 'yellow', 'orange', 'blue', 'pink']

    for df in dfs:
        clean_time(df)
        get_basics(df)
        get_thrust(df)
        get_ds(df)
        print(df)
        print(rp.summary_cont(df['dV'], decimals=10))

    print(dfs[4])

    dfss = [split_victoria(dfs[7]), split_gregarium(dfs[9]), split_aurita(dfs[10]),
            split_capillata(dfs[11]) + split_capillata2(dfs[12]), split_digitale(dfs[0]),
            split_sp(dfs[1]), split_flavicirrata(dfs[2])]
    # split_cellularia(dfs[8]), split_meleagris(dfs[3]),
    # split_unguiculata(dfs[4]), split_tetraphylla(dfs[5]) + split_tetraphylla2(dfs[6]),

    for df in dfss[3]:
        for row in df.index:
            if df.at[row, 'V'] > 1e-6:
                df.drop(row, inplace=True)

    dfs_count = 0
    fig, ax = plt.subplots()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    total_x = np.array([])
    total_y = np.array([])
    for dfs in dfss:
        x = np.array([])
        y = np.array([])

        for df in dfs:
            x = np.append(x, df["dV"].values)
            y = np.append(y, (-1) * df["dS"].values)

        print(dfs_count)
        plt.scatter(x, y, color=colors[dfs_count], label=name[dfs_count])

        total_x = np.append(total_x, x)
        total_y = np.append(total_y, y)

        dfs_count += 1

    total_x2 = sm.add_constant(total_x)
    mod = sm.OLS(total_y, total_x2)
    fii = mod.fit()
    p_value = fii.summary2().tables[1]['P>|t|']
    print(fii.summary())
    r = np.corrcoef(total_x, total_y)
    polymodel = np.poly1d(np.polyfit(total_x, total_y, 1))
    m, b = polymodel
    polyline = np.linspace(min(total_x), max(total_x), 100)
    plt.plot(polyline, polymodel(polyline), color='black')
    # chisq = sum(np.power((total_y - polymodel2(total_x)), 2)/polymodel2(total_x))
    figtext(0.18, 0.3, "$r^2$: %.2f" % np.power(r[0, 1], 2))
    figtext(0.18, 0.26, "y: %.2f x + %.2E" % (m, b))
    figtext(0.18, 0.22, "p: %.2E" % p_value[1])
    plt.xlabel("Change of Bell Volume $(m^3)$")
    plt.ylabel("Change of Wake Volume $(m^3)$")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


    fig_label = ['no need', 'nope']
    obl_done = 4
    dfs_count = 0
    for i in range(2):
        fig, ax = plt.subplots()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        total_x = np.array([])
        total_y = np.array([])
        for dfs in dfss:
            if obl_done != 0 and dfs_count > 3:
                obl_done -= 1
                continue
            x = np.array([])
            y = np.array([])
            for df in dfs:
                x = np.append(x, df["dV"].values)
                y = np.append(y, df["dS"].values)

            plt.scatter(x, y, color=colors[dfs_count], label=name[dfs_count])

            total_x = np.append(total_x, x)
            total_y = np.append(total_y, y)

            dfs_count += 1
            if dfs_count == 4:
                break

        total_x2 = sm.add_constant(total_x)
        mod = sm.OLS(total_y, total_x2)
        fii = mod.fit()
        p_value = fii.summary2().tables[1]['P>|t|']
        print(fii.summary())
        r = np.corrcoef(total_x, total_y)
        polymodel = np.poly1d(np.polyfit(total_x, total_y, 1))
        m, b = polymodel
        polyline = np.linspace(min(total_x), max(total_x), 100)

        plt.plot(polyline, polymodel(polyline), color='black')
        # chisq = sum(np.power((total_y - polymodel2(total_x)), 2)/polymodel2(total_x))
        if dfs_count > 3:
            figtext(0.15, 0.83, fig_label[1], size=20)
        else:
            figtext(0.15, 0.83, fig_label[0], size=20)
        figtext(0.15, 0.76, "$r^2$: %.2f" % np.power(r[0, 1], 2))
        figtext(0.15, 0.72, "y: %.2f x + %.2E" % (m, b))
        figtext(0.15, 0.68, "p: %.2E" % p_value[1])
        plt.xlabel("Change of Bell Volume $(m^3)$")
        plt.ylabel("Change of Subumbrellar Volume $(m^3)$")
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.show()

    fig_label = ['B', 'C']
    for i in range(2):
        dfs_count = 0
        obl_done = 4
        fig, ax = plt.subplots()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        total_x = np.array([])
        total_y = np.array([])
        for dfs in dfss:
            x = np.array([])
            y = np.array([])

            for df in dfs:
                if i == 0:
                    x = np.append(x, df.loc[df['dV'] >= 0, 'dV'].values)
                    y = np.append(y, (-1) * df.loc[df['dV'] >= 0, 'dS'].values)
                if i == 1:
                    x = np.append(x, df.loc[df['dV'] <= 0, 'dV'].values)
                    if obl_done != 0:
                        y = np.append(y, (-3) * df.loc[df['dV'] <= 0, 'dS'].values)
                        obl_done -= 1
                    else:
                        y = np.append(y, (-1) * df.loc[df['dV'] <= 0, 'dS'].values)

            plt.scatter(x, y, color=colors[dfs_count], label=name[dfs_count])

            total_x = np.append(total_x, x)
            total_y = np.append(total_y, y)

            dfs_count += 1

        total_x2 = sm.add_constant(total_x)
        mod = sm.OLS(total_y, total_x2)
        fii = mod.fit()
        p_value = fii.summary2().tables[1]['P>|t|']
        print(fii.summary())
        r = np.corrcoef(total_x, total_y)
        polymodel = np.poly1d(np.polyfit(total_x, total_y, 1))
        m, b = polymodel
        polyline = np.linspace(min(total_x), max(total_x), 100)

        plt.plot(polyline, polymodel(polyline), color='black')
        # chisq = sum(np.power((total_y - polymodel2(total_x)), 2)/polymodel2(total_x))
        figtext(0.15, 0.3, "$r^2$: %.2f" % np.power(r[0, 1], 2))
        figtext(0.15, 0.26, "y: %.2f x + %.2E" % (m, b))
        figtext(0.15, 0.22, "p: %.2E" % p_value[1])
        figtext(0.15, 0.18, fig_label[i], size=20)
        plt.xlabel("Change of Bell Volume $(m^3)$")
        plt.ylabel("Change of Subumbrellar Volume $(m^3)$")
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

main()

# for df in dfs:
#     if i == 0:
#         x = np.append(x, df.loc[df['dV'] >= 0, 'dV'].values)
#         y = np.append(y, df.loc[df['dV'] >= 0, 'dS'].values)
#     if i == 1:
#         x = np.append(x, df.loc[df['dV'] <= 0, 'dV'].values)
#         if pro_done != 0:
#             y = np.append(y, df.loc[df['dV'] <= 0, 'dS'].values)
#             pro_done -= 1
#         else:
#             y = np.append(y, 3 * df.loc[df['dV'] <= 0, 'dS'].values)

# if pro_done != 0:
#     for df in dfs:
#         if i == 0:
#             x = np.append(x, df.loc[df['dV'] >= 0, 'dV'].values)
#             y = np.append(y, df.loc[df['dV'] >= 0, 'dS'].values)
#         if i == 1:
#             x = np.append(x, df.loc[df['dV'] <= 0, 'dV'].values)
#             y = np.append(y, df.loc[df['dV'] <= 0, 'dS'].values)
#     pro_done -= 1
#
# else:
#     for df in dfs:
#         if i == 0:
#             x = np.append(x, df.loc[df['dV'] >= 0, 'dV'].values)
#             y = np.append(y, df.loc[df['dV'] >= 0, 'dS'].values)
#         if i == 1:
#             x = np.append(x, df.loc[df['dV'] <= 0, 'dV'].values)
#             y = np.append(y, 3 * df.loc[df['dV'] <= 0, 'dS'].values)