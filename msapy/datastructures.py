from typing import Optional
import pandas as pd

# TODO: Update Code To Use Custom Accessors Instead: https://pandas.pydata.org/docs/development/extending.html#registering-custom-accessors


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


class ShapleyTableND(pd.DataFrame):
    _metadata = ["_shape"]

    def __init__(self, dataFrame: pd.DataFrame, shape: Optional[list] = None):
        super().__init__(dataFrame)
        self._shape = shape

    @property
    def _constructor(self):
        return ShapleyTableND

    @property
    def contribution_type(self):
        return "nd"

    @classmethod
    def from_ndarray(cls, shapley_table, columns):
        num_permutation, num_nodes = shapley_table.shape[:2]
        contrib_shape = shapley_table.shape[2:]
        data = shapley_table.reshape(num_permutation, num_nodes, -1)
        mode_size = data.shape[2]
        data = data.transpose((0, 2, 1)).reshape((-1, num_nodes))

        shapley_table = pd.DataFrame(data=data,
                                     index=pd.MultiIndex.from_product(
                                         [range(num_permutation), range(mode_size)], names=[None, "mode_size"]),
                                     columns=columns
                                     )
        shapley_table.index.names = [None, "ND"]
        return cls(shapley_table, contrib_shape)

    @property
    def shapley_modes(self):
        return ShapleyModeND(self.groupby(level=1).mean(), self._shape)
    
    @property
    def _constructor_sliced(self):
        return pd.Series


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
