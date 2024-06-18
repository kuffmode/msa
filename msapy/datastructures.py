import pandas as pd
from typing import Optional

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

    def save_as_csv(self, filepath: str):
        self.to_csv(filepath, index=False)

    @classmethod
    def from_csv(cls, filepath: str):
        df = pd.read_csv(filepath)
        return cls(df)


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

        shapley_table = pd.DataFrame(
            data=data,
            index=pd.MultiIndex.from_product(
                [range(num_permutation), range(mode_size)], names=[None, "mode_size"]
            ),
            columns=columns,
        )
        shapley_table.index.names = [None, "ND"]
        return cls(shapley_table, contrib_shape)

    @property
    def shapley_modes(self):
        return ShapleyModeND(self.groupby(level=1).mean(), self._shape)

    @property
    def _constructor_sliced(self):
        return pd.Series

    def save_as_csv(self, filepath: str):
        shape_str = ",".join(map(str, self._shape))
        with open(filepath, "w") as f:
            f.write(f"#shape={shape_str}\n")
        self.to_csv(filepath, mode="a", index=True)

    @classmethod
    def from_csv(cls, filepath: str):
        with open(filepath, "r") as f:
            first_line = f.readline().strip()
            if first_line.startswith("#shape="):
                shape_str = first_line[len("#shape=") :]
                shape = list(map(int, shape_str.split(","))) if shape_str else None
            else:
                raise ValueError("Invalid file format: missing shape information")
        df = pd.read_csv(filepath, index_col=[0, 1], skiprows=1)
        return cls(df, shape)


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

    def save_as_csv(self, filepath: str):
        shape_str = ",".join(map(str, self._shape))
        with open(filepath, "w") as f:
            f.write(f"#shape={shape_str}\n")
        self.to_csv(filepath, mode="a", index=True)

    @classmethod
    def from_csv(cls, filepath: str):
        with open(filepath, "r") as f:
            first_line = f.readline().strip()
            if first_line.startswith("#shape="):
                shape_str = first_line[len("#shape=") :]
                shape = list(map(int, shape_str.split(","))) if shape_str else None
            else:
                raise ValueError("Invalid file format: missing shape information")
        df = pd.read_csv(filepath, index_col=[0], skiprows=1)
        return cls(df, shape)
