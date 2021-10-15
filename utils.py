from typeguard import typechecked
from typing import Any, Generator, Iterable, Callable, Optional, Dict
from ordered_set import OrderedSet
from joblib import Parallel, delayed
import pandas as pd

@typechecked
def generatorize(*, to_iterate: Iterable[Any]) -> Generator[Any, None, None]:
    for stuff in to_iterate:
        yield stuff


@typechecked
def make_complement_space(*,
                          combination_space: OrderedSet,
                          elements: list) -> OrderedSet:
    elements = frozenset(elements)
    complement_space = OrderedSet()
    for combination in generatorize(to_iterate=combination_space):
        complement_space.add(tuple(elements.difference(combination)))
    return complement_space


@typechecked
def parallelized_take_contributions(*,
                                    complement_space: OrderedSet,
                                    combination_space: OrderedSet,
                                    objective_function: Callable,
                                    objective_function_params: Optional[Dict] = None) -> Dict:
    objective_function_params = objective_function_params if objective_function_params else {}

    results = (Parallel(n_jobs=-1)(delayed(objective_function)(
        complement, **objective_function_params) for complement in generatorize(
        to_iterate=complement_space)))

    keys = [frozenset(key) for key in combination_space.items]
    contributions = dict(zip(combination_space, results))
    return contributions


@typechecked
def sorter(shapley_table:pd.DataFrame) -> pd.DataFrame:
    return shapley_table.reindex(shapley_table.mean().sort_values().index, axis=1)
