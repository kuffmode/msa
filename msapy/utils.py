import gc
import warnings
from typing import Any, Generator, Iterable, Callable, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import ray
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor
from ordered_set import OrderedSet
from tqdm import tqdm
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


def ray_iterator(obj_ids):
    """
    Not sure yet what's happening here! I took it for the progress bar from the link below:
    https://github.com/ray-project/ray/issues/5554#issuecomment-558397627
    """
    while obj_ids:
        done, obj_ids = ray.wait(obj_ids)
        yield ray.get(done[0])


@typechecked
def parallelized_take_contributions(*,
                                    multiprocessing_method: str = 'joblib',
                                    n_cores: int = -1,
                                    complement_space: OrderedSet,
                                    combination_space: OrderedSet,
                                    objective_function: Callable,
                                    objective_function_params: Optional[Dict] = None) -> Tuple[Dict, Dict]:
    """
    Same as the take_contribution function but parallelized over CPU cores to boost performance.
    I'd first try the single msapy version on a toy example to make sure everything makes sense then
    go for this because debugging parallel jobs is a disaster. Also, you don't need this if your game
    is happening on GPU. For HPC systems, I guess either dask or ray will be better options.
    ---------------
    Note on returns:
        Contributions and lesion effects are virtually the same thing it's just about how you're looking at them.
        For example, you might want to use lesion effects by conditioning elements' length and see the effect of
        single lesions, dual, triple,... so, for contributions we have a value contributed by the intact coalition,
        the same result can be compared to the intact system to see how big was the impact of lesioning the complements.
        "Same same, but different, but still same!" - James Franco

    Args:
        multiprocessing_method (str):
            So far, two methods of parallelization is implemented, 'joblib' and 'ray' and the default method is joblib.
            If using ray tho, you need to decorate your objective function with @ray.remote decorator. Visit their
            documentations to see how to go for it. I guess ray works better on HPC clusters (if they support it tho!)
            and probably doesn't suffer from the sneaky "memory leakage" of joblib. But just by playing around,
            I realized joblib is faster for tasks that are small themselves. Remedies are here:
            https://docs.ray.io/en/latest/auto_examples/tips-for-first-time.html

            Note: Generally, multiprocessing isn't always faster as explained above. Use it when the function itself
            takes some like each game takes longer than 0.5 seconds or so. For example, a function that sleeps for a
            second on a set of 10 elements with 1000 permutations each (1024 games) performs as follows:

                - no parallel: 1020 sec
                - joblib: 63 sec
                - ray: 65 sec

            That makes sense since I have 16 cores and 1000/16 is around 62.
        # TODO: allow more flexibility in ray method. Scaling up to a cluster?

        n_cores (int):
            Number of parallel games. Default is -1, which means all cores so it can make the system
            freeze for a short period, if that happened then maybe go for -2, which means one msapy is
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

    Returns (Tuple[Dict, Dict]):
        contributions: A dictionary of coalitions:results
        lesion_effects: A dictionary of lesions:results
    """
    objective_function_params = objective_function_params if objective_function_params else {}

    if len(complement_space.items[0]) == 1:
        warnings.warn("Are you sure you're not mistaking complement and combination spaces?"
                      "Length of the first element in complement space is 1, that is usually n_elements-1",
                      stacklevel=2)
    if multiprocessing_method == 'ray':
        if type(objective_function) is not ray.remote_function.RemoteFunction:
            raise ValueError("Objective function is not decorated with ray. You probably forgot @ray.remote")

        if n_cores <= 0:
            warnings.warn("A zero or a negative n_cores was passed and ray doesn't like so "
                          "to fix that ray.init() will get no arguments, "
                          "which means use all cores as n_cores = -1 does for joblib.", stacklevel=2)
            ray.init()
        else:
            ray.init(num_cpus=n_cores)

        result_ids = [objective_function.remote(complement, **objective_function_params) for complement in generatorize(
            to_iterate=complement_space)]
        for _ in tqdm(ray_iterator(result_ids), total=len(result_ids)):
            pass

        results = ray.get(result_ids)
        ray.shutdown()

    elif multiprocessing_method == 'joblib':
        if type(objective_function) is ray.remote_function.RemoteFunction:
            raise ValueError('The objective function is decorated with ray. '
                             'Either comment @ray.remote or use ray as the multiprocessing_method')

        results = (Parallel(n_jobs=n_cores)(delayed(objective_function)(
            complement, **objective_function_params) for complement in tqdm(generatorize(
            to_iterate=complement_space), total=len(complement_space), desc='Playing the games: ')))
    else:
        raise NotImplemented("Available multiprocessing backends are 'ray' and 'joblib'")

    contributions = dict(zip(combination_space, results))
    lesion_effects = dict(zip(complement_space, results))

    gc.collect()
    get_reusable_executor().shutdown(wait=True)

    return contributions, lesion_effects


@typechecked
def distribution_of_processing(*, shapley_vector: pd.Series) -> np.float64:
    """
    Calculates how much the function is distributed accross the system, with values close to 0 means more localized
    functions and values near 1 means most elements are fairly involved in producing the outcome. Remember, this value
    will be low if many units have near zero shapley values while a few has either positive or negative contributions.
    So, negative contributions still count as involvment in the process.

    read more here:
        Aharonov, R., Segev, L., Meilijson, I., & Ruppin, E. 2003.
        Localization of function via lesion analysis.
        Neural Computation.

    and here:
        Saggie-Wexler, Keren, Alon Keinan, and Eytan Ruppin. 2006.
        Neural Processing of Counting in Evolved Spiking and McCulloch-Pitts Agents.
        Artificial Life.

    Args:
        shapley_vector (pd.DataFrame):
            Shapley values of the system, not the shapley table tho, shapley values themselves, i.e., averaged over
            samples so each element has one shapley value.

    returns (int):
        d, distribution of processing!
    """
    normalized = shapley_vector / shapley_vector.abs().sum()  # L1 norm
    d = 1 - normalized.std(ddof=0) / np.sqrt((len(normalized) - 1) / len(normalized) ** 2)
    return d


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
