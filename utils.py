import gc
from typing import Any, Generator, Iterable, Callable, Optional, Dict

import pandas as pd
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor
from ordered_set import OrderedSet
from typeguard import typechecked


@typechecked
def generatorize(to_iterate: Iterable[Any]) -> Generator[Any, None, None]:
    """
    Not sure how useful is this but this function makes a generator out of an iterable.
    Probably good for very large combinaton_spaces.
    Args:
        to_iterate (Iterable[Any]):
            The thing to iterate!

    returns (Generator[Any]):
        It actually yields an iteration but yeah.
    """
    for stuff in to_iterate:
        yield stuff


@typechecked
def make_complement_space(*,
                          combination_space: OrderedSet,
                          elements: list) -> OrderedSet:
    """
    Produces the complement space of the combination space, useful for debugging
    and the multiprocessing function.
    Args:
        combination_space (OrderedSet):
            ordered set of target combinations (coalitions).
        elements (list):
            list of players.

    returns (OrderedSet):
        complements to be passed for lesioning.
    """
    elements = frozenset(elements)
    complement_space = OrderedSet()
    for combination in generatorize(to_iterate=combination_space):
        complement_space.add(tuple(elements.difference(combination)))
    return complement_space


@typechecked
def parallelized_take_contributions(*,
                                    n_cores: int = -1,
                                    complement_space: OrderedSet,
                                    combination_space: OrderedSet,
                                    objective_function: Callable,
                                    objective_function_params: Optional[Dict] = None) -> Dict:
    """
    Same as the take_contribution function but parallelized over CPU cores to boost performance.
    I'd first try the single core version on a toy example to make sure everything makes sense then
    go for this because debugging parallel jobs is a disaster. Also, you don't need this if your game
    is happening on GPU. For HPC systems, I guess either dask or ray will be better options.
    # TODO: compatibility with GPU and HPC.
    Args:
        n_cores (int):
            Number of parallel games. Default is -1, which means all cores so it can make the system
            freeze for a short period, if that happened then maybe go for -2, which means one core is
            left out. Or really just specify the number of threads you want to use!

        complement_space (OrderedSet):
            The actual targets for lesioning. Shapley values are the added contributions of elements
            while in MSA we calculate them by perturbation so although it's intuitive to think the combination
            in combination space is the element that will be lesioned, it is not the case,
            it will be everything else but the coalition, i.e., the target coalition are the only intact elements.

        combination_space (OrderedSet):
            The template, will be copied, filled by the objective_function, and returned.

        objective_function (Callable):
            The game, it should get the complement set and return one numeric value either int or float.
            This function is just calling it as: objective_function(complement, **objective_function_params)
            so design accordingly.

            An example using networkx with some tips:
            (you sometimes need to specify what should happen during edge-cases like an all-lesioned network)

            def local_efficiency(complements, graph):
                if len(complements) < 0:
                    # the network is intact so:
                    return nx.local_efficiency(graph)

                elif len(complements) == len(graph):
                    # the network is fully lesioned so:
                    return 0.0

                else:
                    # lesion the system, calculate things
                    lesioned = graph.copy()
                    lesioned.remove_nodes_from(complements)
                    return nx.local_efficiency(lesioned)

        objective_function_params (Optional[Dict]):
            Kwargs for the objective_function.

    Returns (Dict):
        A dictionary of combinations:results
    """
    objective_function_params = objective_function_params if objective_function_params else {}

    results = (Parallel(n_jobs=n_cores)(delayed(objective_function)(
        complement, **objective_function_params) for complement in generatorize(
        to_iterate=complement_space)))

    contributions = dict(zip(combination_space, results))

    gc.collect()
    get_reusable_executor().shutdown(wait=True)
    return contributions


@typechecked
def sorter(shapley_table: pd.DataFrame) -> pd.DataFrame:
    """
    Sorts the elements based on their average shapley values.
    Args:
        shapley_table(pd.DataFrame):
        unsorted shapley table.

    Returns (pd.DataFrame):
        sorted shapley table.
    """
    return shapley_table.reindex(shapley_table.mean().sort_values().index, axis=1)
