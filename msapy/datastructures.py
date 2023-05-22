from functools import cached_property
from typing import Optional
from msapy import utils as ut, plottings as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#TODO: Update Code To Use Custom Accessors Instead: https://pandas.pydata.org/docs/development/extending.html#registering-custom-accessors

class ShapleyTable(pd.DataFrame):
    @property
    def _constructor(self):
        return ShapleyTable

    @property
    def contribution_type(self):
        return "scaler"

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


class ShapleyModeND(pd.DataFrame):
    _metadata = ["_shape"]

    def __init__(self, dataFrame: pd.DataFrame, shape: Optional[list] = None):
        super().__init__(dataFrame)
        self._shape = shape
    
    @property
    def contribution_type(self):
        return "nd"

    @property
    def _constructor(self):
        return ShapleyModeND

    def get_shapley_mode(self, element):
        return self[element].values.reshape(self._shape)

    def get_total_contributions(self):
        return self.values.sum(1).reshape(self._shape)
