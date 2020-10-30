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

    # import pre-cleaned up csv files, 1 prolate 1 oblate
    s_sp = pd.read_csv(s_sp_f)
    p_gregarium = pd.read_csv(p_gregarium_f)
    # group the two medusae of interest for easy access
    dfs = [s_sp, p_gregarium]
    name = ['S. sp', 'P. gregarium']

    # s_sp: 0.85cm diameter, p_gregarium: 2.14cm diameter (Colin & Costello, 2001)
    s_sp_ori = ori(0.85 / 100)
    p_gregarium_ori = ori(2.14 / 100)
    # group the two constant orifice areas for easy access
    oris = [s_sp_ori, p_gregarium_ori]

    for (df, o) in zip(dfs, oris):
        clean_time(df)
        get_basics(df)
        get_dsdt(df)
        sub_n_vol_change(df, o)
        print(df)
        print(rp.summary_cont(df['dV'], decimals=10))

    dfss = [split_sp(dfs[0]), split_gregarium(dfs[1])]

    dfs_count = 0
    for dfs in dfss:
        total_x = np.array([])
        total_y = np.array([])
        for df in dfs:
            total_x = np.append(total_x, df["dV"].values)
            total_y = np.append(total_y, df["dS"].values)
        total_x2 = sm.add_constant(total_x)
        mod = sm.OLS(total_y, total_x2)
        fii = mod.fit()
        p_value = fii.summary2().tables[1]['P>|t|']
        print(fii.summary())
        plt.scatter(total_x, total_y)
        r = np.corrcoef(total_x, total_y)
        polymodel = np.poly1d(np.polyfit(total_x, total_y, 1))
        m, b = polymodel
        polyline = np.linspace(min(total_x), max(total_x), 100)
        plt.plot(polyline, polymodel(polyline), label='dV to dS linear regression')
        plt.title("%s change of subumbrellar volume with respect to change of bell volume" % (name[dfs_count]))
        # chisq = sum(np.power((total_y - polymodel2(total_x)), 2)/polymodel2(total_x))
        figtext(0.15, 0.8, "r^2: %.2f" % np.power(r[0, 1], 2))
        figtext(0.15, 0.76, "line: %.3f x + %.2E" % (m, b))
        figtext(0.15, 0.72, "constant p_value: %.3E" % p_value[0])
        figtext(0.15, 0.68, "x p_value: %.3E" % p_value[1])
        plt.legend()
        plt.xlabel("Change of Bell Volume m^3")
        plt.ylabel("Change of Subumbrellar Volume m^3")
        plt.tight_layout()
        plt.show()
        dfs_count += 1

main()
