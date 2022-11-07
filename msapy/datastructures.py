from re import S
from typing import Optional
from msapy import utils as ut, plottings as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class ShapleyTable(pd.DataFrame):
    @property
    def _constructor(self):
        return ShapleyTable

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
    def _constructor(self):
        return ShapleyTableMultiScores

    @property
    def shapley_values(self):
        return self.groupby(level=0).mean()

    def get_shapley_table_score(self, score):
        return ShapleyTable(self.loc[score])

    @property
    def scores(self):
        return list(self.index.levels[0])


class ShapleyTableNDBaseClass(pd.DataFrame):
    @property
    def _constructor(self):
        return ShapleyTableNDBaseClass

    @classmethod
    def from_dataframe(cls, shapley_table):
        num_permutation, num_nodes = shapley_table.shape
        data = np.stack(shapley_table.values.flatten())
        mode_size = data.shape[-1]
        data = data.reshape(num_permutation, num_nodes, -1)
        data = data.transpose((0, 2, 1)).reshape((-1, num_nodes))

        shapley_table = pd.DataFrame(data=data,
                                     index=pd.MultiIndex.from_product([range(num_permutation), range(mode_size)], names=[None, "mode_size"]),
                                     columns=shapley_table.columns
                                     )
        return cls(shapley_table)

    @property
    def shapley_modes(self):
        return self.groupby(level=1).mean()

class ShapleyTableTimeSeries(ShapleyTableNDBaseClass):
    @property
    def _constructor(self):
        return ShapleyTableTimeSeries

    @classmethod
    def from_dataframe(cls, shapley_table):
        shapley_table = super(ShapleyTableTimeSeries, cls).from_dataframe(shapley_table)
        shapley_table.index.names = [None, "timestamps"]
        return cls(shapley_table)

    def plot_total_contributions(self, dpi=100, xlabel="Time steps", ylabel="Contribution", title="Total Contributions", savepath=None):
        plt.figure(dpi=dpi)
        plt.plot(self.shapley_modes.sum(1), lw=4.5)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if savepath:
            plt.savefig(savepath, dpi=dpi, bbox_inches='tight')

class ShapleyTableND(ShapleyTableNDBaseClass):
    _metadata = ["_shape"]

    def __init__(self, dataFrame: pd.DataFrame, shape: Optional[list] = None):
        super().__init__(dataFrame)
        self._shape = shape


    @property
    def _constructor(self):
        return ShapleyTableND

    @classmethod
    def from_dataframe(cls, shapley_table, shape):
        shapley_table = ShapleyTableNDBaseClass.from_dataframe(shapley_table)
        shapley_table.index.names = [None, "pixel_id"]
        var =  cls(shapley_table, shape)
        return var

    def get_shapley_mode(self, element):
        return self.shapley_modes[element].values.reshape(self._shape)

