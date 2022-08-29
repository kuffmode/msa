from msapy import utils as ut, plottings as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class ShapleyTable(pd.DataFrame):
    @property
    def shapley_values(self):
        return self.mean()

    def plot_shapley_ranks(self, dpi=100, xlabel="", ylabel="", title="", savepath=None):
        # sorting based on the average contribution (Shapley values)
        shapley_table = ut.sorter(self)

        fig, ax = plt.subplots()
        colors = pl.color_code(shapley_table=shapley_table)
        pl.plot_shapley_ranks(shapley_table=shapley_table,
                              colors=colors, ax=ax)
        fig.set_dpi(dpi)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)

        if savepath:
            plt.savefig(savepath, dpi=dpi, bbox_inches='tight')


class ShapleyTableMultiScores(pd.DataFrame):
    @property
    def shapley_values(self, score=None):
        return self.groupby(level=0).mean()

    def get_shapley_table_score(self, score):
        return ShapleyTable(self.loc[score])

    @property
    def scores(self):
        return list(self.index.levels[0])


class ShapleyTableTimeSeries(pd.DataFrame):
    @classmethod
    def from_dataframe(cls, shapley_table):
        num_permutation, num_nodes = shapley_table.shape
        data = np.stack(shapley_table.values.flatten())
        num_timestamps = data.shape[-1]
        data = data.reshape(num_permutation, num_nodes, -1)
        data = data.transpose((0, 2, 1)).reshape((-1, num_nodes))

        shapley_table = pd.DataFrame(data=data,
                                    index=pd.MultiIndex.from_product([range(num_permutation), range(num_timestamps)], names=[None, "timestamp"]),
                                    columns=shapley_table.columns
                                    )

        return cls(shapley_table)

    @property
    def shapley_modes(self):
        return self.groupby(level=1).mean()

    def plot_total_contributions(self, dpi=100, xlabel="Time steps", ylabel="Contribution", title="Total Contributions", savepath=None):
        plt.figure(dpi=dpi)
        plt.plot(self.shapley_modes.sum(1), lw=4.5)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if savepath:
            plt.savefig(savepath, dpi=dpi, bbox_inches='tight')
