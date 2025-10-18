import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    df = pd.read_csv("data.csv")
    df['MSE'] = df['MSE']**(1/2) * 100
    df['Chi2pvalue'] = df['Chi2pvalue'].fillna(0)
    meandf = df.groupby(['n', 'N'], as_index=False).agg({
          'MSE': 'mean',
          'Chi2pvalue': 'min',
      })
    N_ = (
    [x for x in range(5, 51, 5)],
    [x for x in range(60, 101, 10)],
    [x for x in range(150, 501, 50)],
    [x for x in range(1000, 5000, 500)],
    [x for x in range(5000, 10001, 500)]
    )
    n_ = (
        [x for x in range(5, 50, 5)],
        [x for x in range(50, 100, 5)]
    )

    for n_val, group in meandf.groupby('n'):
        plt.plot(group['N'], group['MSE'], marker='o', label=f"Number of levels = {n_val}")
        plt.title(f'RMSE*100 vs Number of balls | Number of levels == {n_val}')
        plt.xlabel('Number of balls')
        plt.ylabel('RMSE*100')
        # plt.legend(title='Number of levels',
        # bbox_to_anchor=(1.05, 1),
        # loc='upper left',
        # borderaxespad=0.)
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(f"./data/{n_val}plot")
        plt.close()

        plt.plot(group['N'], group['Chi2pvalue'], marker='o', label=f"Number of levels = {n_val}")
        plt.title(f'Chi2pvalue vs Number of balls | Number of levels == {n_val}')
        plt.xlabel('Number of balls')
        plt.ylabel('Chi2pvalue')
        # plt.legend(title='Number of levels',
        # bbox_to_anchor=(1.05, 1),
        # loc='upper left',
        # borderaxespad=0.)
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(f"./n_fixedChi/{n_val}nvalue_plot")
        plt.close()

    for N_val, group in meandf.groupby('N'):
        plt.plot(group['n'], group['MSE'], marker='o', label=f"Number of balls = {N_val}")
        plt.title(f'RMSE*100 vs Number of levels | Number of balls == {N_val}')
        plt.xlabel('Number of levels')
        plt.ylabel('RMSE*100')
        # plt.legend(title='Number of balls',
        # bbox_to_anchor=(1.05, 1),
        # loc='upper left',
        # borderaxespad=0.)
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(f"./data/{N_val}plotNfixed")
        plt.close()

        plt.plot(group['n'], group['Chi2pvalue'], marker='o', label=f"Number of balls = {N_val}")
        plt.title(f'Chi2pvalue vs Number of levels | Number of balls == {N_val}')
        plt.xlabel('Number of levels')
        plt.ylabel('Chi2pvalue')
        # plt.legend(title='Number of levels',
        # bbox_to_anchor=(1.05, 1),
        # loc='upper left',
        # borderaxespad=0.)
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(f"./balls_fixedChi/{N_val}Nvalue_plot")
        plt.close()
    

    i = 0
    for Nprime in N_:
        for nprime in n_:
            # print(meandf)
            meandf2 = meandf[meandf['N'].isin(Nprime)]
            meandf2 = meandf2[meandf2['n'].isin(nprime)]
            for n_val, group in meandf2.groupby('n'):
                plt.plot(group['N'], group['MSE'], marker='o', label=f"n = {n_val}")
            plt.title(f'RMSE*100 vs Number of balls')
            plt.xlabel('Number of balls')
            plt.ylabel('RMSE*100')
            plt.legend(title='Number of levels',
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            borderaxespad=0.)
            plt.tight_layout()
            plt.grid(True)
            plt.savefig(f"./plot_together/number_levels_iter{i}")
            plt.close()

            for N_val, group in meandf2.groupby('N'):
                plt.plot(group['n'], group['MSE'], marker='o', label=f"N = {N_val}")
            plt.title(f'RMSE*100 vs Number of levels')
            plt.xlabel('Number of levels')
            plt.ylabel('RMSE*100')
            plt.legend(title='Number of balls',
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            borderaxespad=0.)
            plt.tight_layout()
            plt.grid(True)
            plt.savefig(f"./plot_together/number_balls_iter{i}")
            plt.close()
            
            heatmap_data = meandf2.pivot(index='n', columns='N', values='MSE')

            plt.figure(figsize=(8, 6))
            sns.heatmap(
                heatmap_data,
                annot=True,        # Muestra los valores dentro de cada celda
                fmt=".4f",         # Formato numérico
                cmap='YlOrRd',    # Colormap (puedes cambiarlo: 'coolwarm', 'plasma', etc.)
                cbar_kws={'label': 'RMSE*100'}
            )

            plt.title('Heatmap of RMSE*100 | Number of levels and Number of balls')
            plt.xlabel('Number of balls')
            plt.ylabel('Number of levels')
            plt.tight_layout()
            plt.savefig(f"./plot_together/HeatMap_iter{i}")
            plt.close()

            i += 1