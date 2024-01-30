import importlib
import warnings
from typing import Callable, Optional, Dict, Tuple
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from ordered_set import OrderedSet
from itertools import combinations
from typeguard import typechecked
from tqdm import tqdm
from fastprogress.fastprogress import master_bar, progress_bar, MasterBar
from tqdm_joblib import tqdm_joblib

from msapy import utils as ut
from msapy.checks import _check_get_shapley_table_args, _check_valid_combination_space, _check_valid_elements, _check_valid_n_permutations, _check_valid_permutation_space, _get_contribution_type, _is_number
from msapy.datastructures import ShapleyModeND, ShapleyTable, ShapleyTableND


@typechecked
def make_permutation_space(*,
                           elements: list,
                           n_permutations: int,
                           pair: Optional[Tuple] = None,
                           rng: Optional[np.random.Generator] = None,
                           random_seed: Optional[int] = None) -> list:
    """
    Generates a list of tuples containing n_permutations of the given elements.
    This will be used later in make_combination_space so you can have the same permutation and combination spaces for
    different games played by the same set. Probably makes things more reproducible!
    The elements themselves can be anything I guess, I tried str (names/labels) and integers (indexes),
    and tuples (edges, from-to style).
    Briefly, the permutation space of (A,B,C) is something like this:

    (A,B,C)
    (B,C,A)
    (C,B,A)
    (A,C,B)
    (C,A,B)
    (B,A,C)
    (C,A,B)
    .
    .
    .
    As you might have seen, there might be repetitions for small set of players and that's fine.

    Args:
        elements (list):
            A list of players to be shuffled n times.

        n_permutations (int):
            Number of permutations, Didn't check it systematically yet but just based on random explorations I'd say
            something around 1_000 is enough.

        pair (Optional[Tuple]):
            pair of elements that will always be together in every permutation

        rng (Optional[np.random.Generator]): Numpy random generator object used for reproducable results. Default is None.

        random_seed (Optional[int]):
            sets the random seed of the sampling process. Only used when `rng` is None. Default is None.

    Returns:
        (list[tuple]): Permutation space as a list of lists with shape (n_permutations, len(elements))
    """

    # ------------------------------#
    _check_valid_elements(elements)
    _check_valid_n_permutations(n_permutations)
    # ------------------------------#

    type_of_elements = type(elements[0])

    # create a numpy random number generator if one is not passed
    if not rng:
        rng = np.random.default_rng(random_seed) if random_seed \
            else np.random.default_rng()

    # return n_permutations random permutations if pair argument is not passed
    if not pair:
        permutation_space = [tuple(type_of_elements(element)
                                   for element in rng.permutation(elements))
                             for _ in range(n_permutations)]

        return permutation_space

    # if the pair argument is passed, then all permutations will have those two elements together using the following logic
    elements = [e for e in elements if e != pair[0]]
    permutation_space = []

    for _ in range(n_permutations):
        permutation = list(rng.permutation(elements))
        permutation.insert(permutation.index(pair[1]), pair[0])
        permutation_space.append(tuple(permutation))

    return permutation_space


@typechecked
def make_combination_space(*, permutation_space: list, pair: Optional[Tuple] = None, lesioned: Optional[any] = None) -> OrderedSet:
    """
    Generates a dataset (OrderedSet) of coalitions from the permutation_space.
    In principle, this could be directly filled and passed to the make_shapley_values function
    but then the function wouldn't be pure so **this will be just an empty template**.
    Don't mix up this and the later-filled combination space.
    Briefly, the combination space of **one permutation of** (A,B,C) is something like this:

    (A,B,C)
    (A,B)
    (B,C)
    (A,C)
    (C)
    (B)
    (A)
    ()

    This will happen for every permutation of the permutation space so either there will be a large number
    of combinations here or if the set is small enough, it will be exhausted.

    Args:
        permutation_space (list):
            A list of players to be shuffled n times.

        pair (Optional[Tuple]):
            pair of elements that will always be together in every combination

        lesioned (Optional[any]):
            leseioned element that will not be present in any combination

    Returns:
        (OrderedSet): Combination space as an OrderedSet of frozensets.
    """

    _check_valid_permutation_space(permutation_space)

    # if we have an element that needs to be lesioned in every combination, then we store it in a set so that taking a difference becomes easier and efficient
    lesioned = {lesioned} if lesioned else set()

    combination_space = OrderedSet()

    # iterate over all permutations and generate including and excluding combinations
    for permutation in permutation_space:
        # we need this parameter if we have a pair so that the pair of elements are always together
        skip_next = False
        for index, element in enumerate(permutation):
            # logic to skip the next element if we encounter a pair element
            if skip_next:
                skip_next = False
                continue
            if pair and element == pair[0]:
                index += 1
                skip_next = True

            # forming the coalition with the target element
            including = frozenset(permutation[:index + 1]) - lesioned
            # forming it without the target element
            excluding = frozenset(permutation[:index]) - lesioned

            combination_space.add(including)
            combination_space.add(excluding)

    return combination_space


@typechecked
def make_complement_space(*,
                          combination_space: OrderedSet,
                          elements: list,
                          lesioned: Optional[any] = None) -> OrderedSet:
    """
    Produces the complement space of the combination space, useful for debugging
    and the multiprocessing function.
    Args:
        combination_space (OrderedSet):
            ordered set of target combinations (coalitions).
        elements (list):
            list of players.
        lesioned (Optional[any]):
            leseioned element that will not be present in any combination but every complement

    Returns:
        (OrderedSet): complements to be passed for lesioning.
    """
    _check_valid_elements(elements)
    elements = frozenset(elements)
    _check_valid_combination_space(combination_space, elements, lesioned)

    complement_space = OrderedSet()

    # iterate over all combinations and take their difference from set elements to find complements
    for combination in combination_space:
        complement_space.add(tuple(elements.difference(combination)))
    return complement_space


@typechecked
def take_contributions(*,
                       elements: list,
                       complement_space: OrderedSet,
                       combination_space: OrderedSet,
                       objective_function: Callable,
                       objective_function_params: Optional[Dict] = None,
                       mbar: Optional[MasterBar] = None) -> Tuple[Dict, Dict]:
    """
    This function fills up the combination_space with the game you define (objective function). There is an important
    point to keep in mind, Shapley values are the added contributions of elements while in MSA we calculate them by
    perturbation so although it's intuitive to think the combination in combination space is the element that will be
    lesioned, it is not the case, it will be everything else but the coalition, i.e., the target coalition are the
    only intact elements. This function takes care of this by passing the complement of each coalition to the
    game while assigning the results to the target coalition, just keep the logic in mind.

    A second point is that this function returns a filled combination_space, it is not filling it in-place for the
    sake of purity.

    ---------------
    Note on returns:
        Contributions and lesion effects are virtually the same thing it's just about how you're looking at them.
        For example, you might want to use lesion effects by conditioning elements' length and see the effect of
        single lesions, dual, triple,... so, for contributions we have a value contributed by the intact coalition,
        the same result can be compared to the intact system to see how big was the impact of lesioning the complements.
        "Same same, but different, but still same!" - James Franco

    Args:
        elements (list):
            List of the players. Obviously, should be the same passed to make_permutation.

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

    Returns:
        (Dict): A dictionary of combinations:results
    """

    elements = frozenset(elements)
    contributions = dict.fromkeys(combination_space)
    lesion_effects = dict.fromkeys(complement_space)
    objective_function_params = objective_function_params if objective_function_params else {}

    # ------------------------------#
    if len(complement_space.items[1]) == 0:
        warnings.warn("Are you sure you're not mistaking complement and combination spaces?"
                      "Length of the first element in complement space (really, complement_space[1]) is 0. "
                      "It should be equal to the number of elements.",
                      stacklevel=2)
    # ------------------------------#

    # run the objective function over all complement space
    for combination, complement in progress_bar(zip(combination_space, complement_space), parent=mbar, total=len(combination_space), leave=False):
        result = objective_function(complement, **objective_function_params)

        contributions[combination] = result
        lesion_effects[complement] = result
    return contributions, lesion_effects


@typechecked
def get_shapley_table(*,
                      permutation_space: list,
                      contributions: Optional[Dict] = None,
                      lesioned: Optional[any] = None,
                      objective_function: Optional[Callable] = None,
                      objective_function_params: Optional[Dict] = None,
                      lazy=False,
                      save_permutations: bool = False,
                      dual_progress_bars: bool = True,
                      mbar: Optional[MasterBar] = None,) -> pd.DataFrame:
    """
    Calculates Shapley values based on the filled contribution_space.
    Briefly, for a permutation (A,B,C) it will be:

    (A,B,C) - (B,C) = Contribution of A to the coalition (B,C).
    (B,C) - (C) = Contribution of B to the coalition formed with (C).
    (C) = Contribution of C alone.

    This will repeat over all permutations. and the result is a distribution of Shapley values for each element,
    note that the estimation method we're using here is an "unbiased estimator" so the variance is fairly large.

    Args:
        contributions (Dict):
            Filled Dictionary of coalition:result

        permutation_space (list):
            Should be the same passed to make_combination_space.

        lesioned (Optional[any]):
            leseioned element that will not be present in any combination

        objective_function (Callable):
            The game (in-silico experiment). It should get the complement set and return one numeric value
            either int or float.
            This function is just calling it as: objective_function(complement, **objective_function_params)

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

        objective_function_params (Dict):
            Kwargs for the objective_function.

        lesioned (Optional[any]):
            leseioned element that will not be present in any combination

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

        rng (Optional[np.random.Generator]): Numpy random generator object used for reproducable results. Default is None.

        random_seed (Optional[int]):
            sets the random seed of the sampling process. Only used when `rng` is None. Default is None.

        n_parallel_games (int):
            Number of parallel jobs (number of to-be-occupied cores),
            -1 means all CPU cores and 1 means a serial process.
            I suggest using 1 for debugging since things get messy in parallel!

        lazy (bool): if set to True, objective function will be called lazily instead of calling it all at once and storing the outputs in a dict.
            Setting it to True saves a lot of memory and might even be faster in certain cases.

        save_permutations (bool): If set to True, the shapley values are calculated by calculating the running mean of the permutations instead of
            storing the permutations. This parameter is ignored in case the objective function returns a scaler.

        dual_progress_bar (bool): If set to true, you will have two progress bars. One parent that will track the permutations, other child that
            will track the elements. Its ignored in case the mbar is provided

        mbar (MasterBar): A Fastprogress MasterBar. Use it in case you're calling the interface multiple times to have a nester progress bar.

    Returns:
        pd.DataFrame: Shapley table or a dict of Shapely tables, columns will be 
        elements and indices will be samples (permutations). 
        It will be a Multi-Index DataFrame if the contributions are a timeseries.
        The index at `level=1` will be the timestamps
    """
    _check_get_shapley_table_args(contributions, objective_function, lazy)
    _check_valid_permutation_space(permutation_space)

    lesioned = {lesioned} if lesioned else set()
    contributions = {tuple(lesioned): objective_function(tuple(lesioned), **objective_function_params)} if lazy else contributions

    contribution_type, intact_contributions_in_case_lazy = _get_contribution_type(contributions)
    contrib_shape = intact_contributions_in_case_lazy.shape if contribution_type == "nd" else []

    sorted_elements = sorted(permutation_space[0])
    permutation_space = set(permutation_space)

    if not lazy:
        parent_bar = enumerate(permutation_space)
    elif (not dual_progress_bars) or mbar:
        parent_bar = progress_bar(enumerate(permutation_space), total=len(
            permutation_space), leave=False, parent=mbar)
    elif lazy:
        parent_bar = master_bar(
            enumerate(permutation_space), total=len(permutation_space))

    shapley_table = 0 if (contribution_type == 'nd' and not save_permutations) else np.zeros((len(permutation_space), len(sorted_elements), *contrib_shape), dtype=float)

    for i, permutation in parent_bar:
        isolated_contributions = np.zeros((len(permutation), *intact_contributions_in_case_lazy.shape), dtype=float) if contribution_type=="nd" else ([None] * len(permutation))  # got to be a better way!
        child_bar = enumerate(permutation) if not (dual_progress_bars and lazy) else progress_bar(
            enumerate(permutation), total=len(permutation), leave=False, parent=parent_bar)
        # iterate over all elements in the permutation to calculate their isolated contributions
        
        contributions_including = intact_contributions_in_case_lazy
        for index, element in child_bar:
            including = frozenset(permutation[:index + 1])
            excluding = frozenset(permutation[:index])

            # the isolated contribution of an element is the difference of contribution with that element and without that element
            if lazy:
                contributions_excluding = objective_function(tuple(including.union(lesioned)), **objective_function_params)
                isolated_contributions[sorted_elements.index(element)] = contributions_including - contributions_excluding
                contributions_including = contributions_excluding
            else:
                isolated_contributions[sorted_elements.index(element)] =  contributions[including - lesioned] - contributions[excluding - lesioned]

        if contribution_type == 'nd' and not save_permutations:
            shapley_table += (isolated_contributions - shapley_table) / (i + 1)
        else:
            shapley_table[i] = np.array(isolated_contributions)

    # post processing of shapley values based on what type of contribution it is. The format of output will vary based on if the
    # values are multi-scores, timeseries, etc.
    if contribution_type == 'nd' and not save_permutations:
        shapley_table = shapley_table.reshape(shapley_table.shape[0], -1).T
        shapley_table = pd.DataFrame(
            shapley_table, columns=sorted_elements)
        return ShapleyModeND(shapley_table, intact_contributions_in_case_lazy.shape)

    if contribution_type == "scaler":
        return ShapleyTable(pd.DataFrame(shapley_table, columns=sorted_elements))
    
    return ShapleyTableND.from_ndarray(shapley_table, columns=sorted_elements)


@typechecked
def interface(*,
              n_permutations: int,
              elements: list,
              objective_function: Callable,
              objective_function_params: Dict = {},
              permutation_space: Optional[list] = None,
              pair: Optional[Tuple] = None,
              lesioned: Optional[any] = None,
              multiprocessing_method: str = 'joblib',
              rng: Optional[np.random.Generator] = None,
              random_seed: Optional[int] = None,
              n_parallel_games: int = -1,
              lazy: bool = True,
              save_permutations: bool = False,
              dual_progress_bars: bool = True,
              mbar: Optional[MasterBar] = None
              ) -> pd.DataFrame:
    """
    A wrapper function to call other related functions internally and produces an easy-to-use pipeline.

    Args:
        n_permutations (int):
            Number of permutations (samples) per element.

        elements (list):
            List of the players (elements). Can be strings (names), integers (indicies), and tuples.

        objective_function (Callable):
            The game (in-silico experiment). It should get the complement set and return one numeric value
            either int or float.
            This function is just calling it as: objective_function(complement, **objective_function_params)

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

        objective_function_params (Dict):
            Kwargs for the objective_function.

        permutation_space (Optional[list]):
            Already generated permutation space, in case you want to be more reproducible or something and use the same
            lesion combinations for many metrics.

        pair (Optional[Tuple]):
            pair of elements that will always be together in every combination

        lesioned (Optional[any]):
            leseioned element that will not be present in any combination

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

        rng (Optional[np.random.Generator]): Numpy random generator object used for reproducable results. Default is None.

        random_seed (Optional[int]):
            sets the random seed of the sampling process. Only used when `rng` is None. Default is None.

        n_parallel_games (int):
            Number of parallel jobs (number of to-be-occupied cores),
            -1 means all CPU cores and 1 means a serial process.
            I suggest using 1 for debugging since things get messy in parallel!

        lazy (bool): if set to True, objective function will be called lazily instead of calling it all at once and storing the outputs in a dict.
            Setting it to True saves a lot of memory and might even be faster in certain cases.

        save_permutations (bool): If set to True, the shapley values are calculated by calculating the running mean of the permutations instead of
            storing the permutations. This parameter is ignored in case the objective function returns a scaler.

        dual_progress_bar (bool): If set to true, you will have two progress bars. One parent that will track the permutations, other child that
            will track the elements. Its ignored in case the mbar is provided

        mbar (MasterBar): A Fastprogress MasterBar. Use it in case you're calling the interface multiple times to have a nester progress bar.


    Returns:
        Tuple[pd.DataFrame, Dict, Dict]: shapley_table, contributions, lesion_effects

    Note that contributions and lesion_effects are the same values, addressed differently. For example:
    If from a set of ABCD removing AC ends with some value x, you can say the contribution of BD=x and the
    effect of removing AC=x. So the same values are addressed differently in the two returned Dicts.
    Of course, it makes more sense to compare the lesion effects with the intact system but who am I to judge.
    """

    # create a numpy random number generator if one is not passed
    if not rng:
        rng = np.random.default_rng(
            random_seed) if random_seed else np.random.default_rng()

    # create a permutation_space if one is not passed
    if not permutation_space:
        permutation_space = make_permutation_space(elements=elements,
                                                   n_permutations=n_permutations,
                                                   pair=pair,
                                                   rng=rng)
    else:
        warnings.warn("A Permutation space is given so n_permutations will fall back to what's specified there.",
                      stacklevel=2)

    if lazy:
        shapley_table = get_shapley_table(permutation_space=permutation_space,
                                          lesioned=lesioned,
                                          lazy=True,
                                          objective_function=objective_function,
                                          objective_function_params=objective_function_params,
                                          dual_progress_bars=dual_progress_bars,
                                          save_permutations=save_permutations,
                                          mbar=mbar)[elements]
        return shapley_table

    combination_space = make_combination_space(permutation_space=permutation_space,
                                               pair=pair,
                                               lesioned=lesioned)
    complement_space = make_complement_space(combination_space=combination_space,
                                             elements=elements,
                                             lesioned=lesioned)

    if n_parallel_games == 1:
        contributions, _ = take_contributions(elements=elements,
                                              complement_space=complement_space,
                                              combination_space=combination_space,
                                              objective_function=objective_function,
                                              objective_function_params=objective_function_params,
                                              mbar=mbar)
    else:
        contributions, _ = ut.parallelized_take_contributions(
            multiprocessing_method=multiprocessing_method,
            n_cores=n_parallel_games,
            complement_space=complement_space,
            combination_space=combination_space,
            objective_function=objective_function,
            objective_function_params=objective_function_params,
            mbar=mbar)

    shapley_table = get_shapley_table(contributions=contributions,
                                      permutation_space=permutation_space,
                                      dual_progress_bars=dual_progress_bars,
                                      save_permutations=save_permutations,
                                      lesioned=lesioned, mbar=mbar)[elements]
    return shapley_table


@typechecked
def interaction_2d(*,
                   n_permutations: int,
                   elements: list,
                   pair: tuple,
                   objective_function: Callable,
                   objective_function_params: Dict = {},
                   multiprocessing_method: str = 'joblib',
                   rng: Optional[np.random.Generator] = None,
                   random_seed: Optional[int] = None,
                   n_parallel_games: int = -1,
                   lazy: bool = False,
                   ) -> Tuple:
    """Performs Two dimensional MSA as explain in section 2.3 of [1]. 
    We calculate the Shapley value of element i in the subgame of all elements without element j. 
    Intuitively, this is the average marginal importance of element i when element j is perturbed. 
    Repeat the process by interchanging i and j and the calculate the shapley values by considering 
    i and j as a single unit.

    REFERENCES:
        Keinan, Alon, et al. "Fair attribution of functional contribution in artificial and biological networks." 
        Neural computation 16.9 (2004): 1887-1915.

    Args:
        n_permutations (int): Number of permutations (samples) per element.

        elements (list): List of the players (elements). Can be strings (names), integers (indicies), and tuples.

        pair (tuple): the pair of elements we want to analyze the interaction between i.e. element i and j

        objective_function (Callable):
            The game (in-silico experiment). It should get the complement set and return one numeric value
            either int or float.
            This function is just calling it as: objective_function(complement, **objective_function_params)

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

        objective_function_params (Dict, optional): Kwargs for the objective_function. Defaults to {}.

        multiprocessing_method (str, optional): 
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
            Defaults to 'joblib'.

        rng (Optional[np.random.Generator], optional): Numpy random generator object used for reproducable results. Default is None. Defaults to None.

        random_seed (Optional[int], optional): 
            sets the random seed of the sampling process. Only used when `rng` is None. Default is None. Defaults to None.

        n_parallel_games (int):
            Number of parallel jobs (number of to-be-occupied cores),
            -1 means all CPU cores and 1 means a serial process.
            I suggest using 1 for debugging since things get messy in parallel!

    Returns:
        tuple: 
            (shapley value of element (i, j), 
            shapley value of element i when j is lesioned, 
            shapley value of element j when i is lesioned) 
    """

    interface_args = {"elements": elements,
                      "objective_function": objective_function,
                      "n_permutations": n_permutations,
                      "objective_function_params": objective_function_params,
                      "multiprocessing_method": multiprocessing_method,
                      "rng": rng,
                      "random_seed": random_seed,
                      "n_parallel_games": n_parallel_games,
                      "save_permutations": False,
                      "lazy": lazy}

    # calculate the shapley values with element j lesioned
    shapley_i = interface(**interface_args, lesioned=pair[1])
    # get the shapley value of element i with element j leasioned
    gamma_i = _get_gamma(shapley_i, [pair[0]]).sum()

    # calculate the shapley values with element i lesioned
    shapley_j = interface(**interface_args, lesioned=pair[0])
    # get the shapley value of element j with element i leasioned
    gamma_j = _get_gamma(shapley_j, [pair[1]]).sum()

    # calculate the shapley values with element i and j together in every combination
    shapley_ij = interface(**interface_args, pair=pair)
    # get the sum of the shapley value of element i and j
    gamma_ij = _get_gamma(shapley_ij, list(pair)).sum()

    return gamma_ij, gamma_i, gamma_j


@typechecked
def network_interaction_2d(*,
                           n_permutations: int,
                           elements: list,
                           pairs: Optional[list] = None,
                           objective_function: Callable,
                           objective_function_params: Dict = {},
                           multiprocessing_method: str = 'joblib',
                           rng: Optional[np.random.Generator] = None,
                           random_seed: Optional[int] = None,
                           n_parallel_games: int = -1,
                           lazy: bool = False
                           ) -> np.ndarray:
    """Performs Two dimensional MSA as explain in section 2.3 of [1]
    for every possible pair of elements and returns a symmetric matrix of
    interactions between the elements.

    Args:
        n_permutations (int): Number of permutations (samplescontributions_excluding) per element.

        elements (list): List of the players (elements). Can be strings (names), integers (indicies), and tuples.

        pairs (Optional[list]): List of pairs of elements that you want to analyze the interaction between. 
            Defaults to None which means all possible pairs

        objective_function (Callable):
            The game (in-silico experiment). It should get the complement set and return one numeric value
            either int or float.
            This function is just calling it as: objective_function(complement, **objective_function_params)

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
contributions_excluding
        objective_function_params (Dict, optional): Kwargs for the objective_function. Defaults to {}.

        multiprocessing_method (str, optional): 
            So far, two methods of parallelization is implemented, 'joblib' and 'ray' and the default method is joblib.
            If using ray tho, you need to decorate your objective function with @ray.remote decorator. Visit their
            documentations to see how to go for it. I guess ray works better on HPC clusters (if they support it tho!)
            and probably doesn't suffer from the sneaky "memory leakage" of joblib. But just by playing around,
            I realized joblib is faster for tasks that are small themselves. Remedies are here:
            https://docs.ray.io/en/latest/auto_examples/tips-for-first-time.html

            Note: Generally, multiprocessing isn't always faster as explained above. Use it when the function itself
            takes some like each game takes longer than 0.5 seconds or so. For example, a function that sleeps for a
            second on a set of 10 elements with 1000 permutations each (1024 games) performs as follows:

                - no parallel: 1020 seccontributions_excluding
                - joblib: 63 sec
                - ray: 65 sec

            That makes sense since I have 16 cores and 1000/16 is around 62. 
            Defaults to 'joblib'.

        rng (Optional[np.random.Generator], optional): Numpy random generator object used for reproducable results. Default is None. Defaults to None.

        random_seed (Optional[int], optional): 
            sets the random seed of the sampling process. Only used when `rng` is None. Default is None. Defaults to None.

        n_parallel_games (int):
            Number of parallel jobs (number of to-be-occupied cores),
            -1 means all CPU cores and 1 means a serial process.
            I suggest using 1 for debugging since things get messy in parallel!


    Raises:
        NotImplementedError: Raises this error in case the contribution is a timeseries or there are
            multiple contributions

    Returns:
        np.ndarray: the interaction matrix
    """
    elements_idx = list(range(len(elements)))

    # create a list of pairs for wich we'll calculate the 2d interaction. By default, all possible pairs are considered unless specified otherwise
    all_pairs = [(elements.index(x), elements.index(y))
                 for x, y in pairs] if pairs else combinations(elements_idx, 2)

    interface_args = {"elements": elements,
                      "n_permutations": n_permutations,
                      "objective_function_params": objective_function_params,
                      "objective_function": objective_function,
                      "multiprocessing_method": multiprocessing_method,
                      "rng": rng,
                      "random_seed": random_seed,
                      "n_parallel_games": n_parallel_games,
                      "lazy": lazy}

    interactions = np.zeros((len(elements), len(elements)))

    # iterate over all the pairs to run interaction_2d
    for x, y in tqdm(all_pairs, desc="Running interface 2d for all pair of nodes:"):
        gammaAB, gammaA, gammaB = interaction_2d(pair=(elements[x], elements[y]),
                                                 **interface_args)
        if not _is_number(gammaAB):
            raise NotImplementedError("`network_interaction_2d` does not work with"
                                      " timeseries or multiscore contributions yet.")
        interactions[x, y] = interactions[y, x] = gammaAB - gammaA - gammaB

    return interactions


def _get_gamma(shapley, idx):
    """returns shapley value of elements in idx

    Args:
        shapley_table (pd.DataFrame): shapley table with one element lesioned
        idx (_type_): element of interest

    Returns:
        shapley value of elements in idx
    """
    if isinstance(shapley, ShapleyTable):
        gamma = shapley.shapley_values[idx]
    elif isinstance(shapley, ShapleyTableND):
        gamma = shapley[idx]
    return gamma


@typechecked
def estimate_causal_influences(elements: list,
                               objective_function: Callable,
                               objective_function_params: Optional[dict] = None,
                               target_elements: Optional[list] = None,
                               multiprocessing_method: str = 'joblib',
                               n_cores: int = -1,
                               n_permutations: int = 1000,
                               permutation_seed: Optional[int] = None,
                               parallelize_over_games=False,
                               lazy=True
                               ) -> pd.DataFrame:
    """
    Estimates the causal contribution (Shapley values) of each node on the rest of the network. Basically, this function
    performs MSA iteratively on each node and tracks the changes in the objective_function of the target node.
    For example we have a chain A -> B -> C, and we want to know how much A and B are contributing to C. We first need to
    define a metric for C (objective_function) which here let's say is the average activity of C. MSA then performs a
    multi-site lesioning analysis of A and B so for each we will end up with a number indicating their contributions to
    the average activity of C.

    VERY IMPORTANT NOTES:

        1. The resulting causal contribution matrix does not necessarily reflect the connectome. In the example above
        there's no actual connection A -> C but there might be one in the causal contribution matrix since A is causally
        influencing C via B.
        2. Think twice (even three times) about your objective function. The same everything will result in different
        causal contribution matrices depending on what are you tracking and how accurate it's capturing the effect of
        lesions. Also don't forget the edge-cases. There will be weird behaviors in your system, for example, what it
        does if every node is perturbed?
        3. The metric you track is preferred to be non-negative and bounded (at least practically!)
        4. Obviously this will take N times longer than a normal MSA with N is the number of nodes. So make sure your
        process is as fast as it can be for example use Numba and stuff, but you don't need to implement any parallel
        processes since it's already implemented here. Going below 1000 permutations might be an option depending on
        your specific case but based on experience, it's not a good idea 
        5. Shapley values sum up (or will be close) to the value of the intact coalition. So for example if the
        mean activity of node C here is 50 then causal_contribution_matrix.sum(axis=0) = 50 or close to 50. If not it
        means:
            1. the number of permutations are not enough
            2. there is randomness somewhere in the process
            3. your objective function is not suitable


    Args:
        elements (list):
            List of the players (elements). Can be strings (names), integers (indicies), and tuples.

        objective_function (Callable):
            The game (in-silico experiment). It should get the complement set and return one numeric value
            either int or float.
            This function is just calling it as: objective_function(complement, **objective_function_params)

            An example using networkx with some tips:

            def lesion_me_senpai(complements, network, index):
                # note "index", your function should be able to track the effects on the target and the keyword for
                  that is "index"

                if len(complements) == len(A)-1:  # -1 since the target node is active
                    return 0

                lesioned_network = deepcopy(network)
                for target in complements:
                    lesioned_network[target] = 0  # setting all connections of the targets to 0

                activity = network.run(lesioned_network) # or really, whatever you want!
                return float(activity[index].mean())

            (you sometimes need to specify what should happen during edge-cases like an all-lesioned network)


        objective_function_params (Optional[Dict]):
            Kwargs for the objective_function. A dictionary pair of {'index': index} will be added to this during
            the process so your function can track the lesion effect.

        target_elements (Optional[list]): list of elements that you want to calculate the causal influence of.

        multiprocessing_method (str = 'joblib'):
            So far, two methods of parallelization is implemented, 'joblib' and 'ray' and the default method is joblib.
            If using ray tho, you need to decorate your objective function with @ray.remote decorator. Visit their
            documentations to see how to go for it.

        n_cores (int = -1):
            Number of parallel games. Default is -1, which means all cores so it can make the system
            freeze for a short period, if that happened then maybe go for -2, which means one msapy is
            left out. Or really just specify the number of threads you want to use!

        n_permutations (int = 1000):
            Number of permutations per node.
            Didn't check it systematically yet but just based on random explorations
            I'd say something around 1000 is enough.

        permutation_seed (Optional[int] = None):
            Sets the random seed of the sampling process. Default is None so if nothing is given every call results in
            a different orderings.

        parallelize_over_games (bool = False): Whether to parallelize over games or parallelize over elements. Parallelizing
            over the elements is generally faster. Defaults to False

    Returns:
        causal_influences (pd.DataFrame)

    """
    target_elements = target_elements if target_elements else elements
    objective_function_params = objective_function_params if objective_function_params else {}

    if parallelize_over_games:
        # run causal_influence_single_element for all target elements.
        mbar = master_bar(enumerate(target_elements),
                          total=len(target_elements))
        results = [causal_influence_single_element(elements, objective_function,
                                                   objective_function_params, n_permutations,
                                                   n_cores, multiprocessing_method,
                                                   permutation_seed, index, element, lazy, mbar) for index, element in mbar]

    elif multiprocessing_method == 'ray':
        if importlib.util.find_spec("ray") is None:
            raise ImportError(
                "The ray package is required to run this algorithm. Install and use at your own risk.")

        import ray
        if n_cores <= 0:
            warnings.warn("A zero or a negative n_cores was passed and ray doesn't like so "
                          "to fix that ray.init() will get no arguments, "
                          "which means use all cores as n_cores = -1 does for joblib.", stacklevel=2)
            ray.init()
        else:
            ray.init(num_cpus=n_cores)

        result_ids = [ray.remote(causal_influence_single_element).remote(elements, objective_function,
                                                                         objective_function_params, n_permutations,
                                                                         1, 'joblib',
                                                                         permutation_seed, index, element, lazy, None) for index, element in enumerate(target_elements)]

        for _ in tqdm(ut.ray_iterator(result_ids), total=len(result_ids)):
            pass

        results = ray.get(result_ids)
        ray.shutdown()

    else:
        with tqdm_joblib(desc="Doing Nodes: ", total=len(target_elements)) as pb:
            results = (Parallel(n_jobs=n_cores)(delayed(causal_influence_single_element)(elements, objective_function,
                                                                                         objective_function_params, n_permutations,
                                                                                         1, 'joblib',
                                                                                         permutation_seed, index, element, lazy) for index, element in enumerate(target_elements)))

    _, contribution_type = results[0]
    shapley_values = [r[0] for r in results]

    causal_influences = pd.DataFrame(
        shapley_values, columns=elements) if contribution_type == "scaler" else pd.concat(shapley_values, keys=elements)

    if contribution_type == "scaler":
        return causal_influences
    return causal_influences[causal_influences.index.levels[0]]


def causal_influence_single_element(elements, objective_function,
                                    objective_function_params, n_permutations,
                                    n_parallel_games, multiprocessing_method,
                                    permutation_seed, index, element, lazy=True, mbar=None):
    """
    Estimates the causal contribution (Shapley values) of a node on the rest of the network. Basically, this function
    performs MSA and tracks the changes in the objective_function of the target node.

    Args:
        elements (list):
            List of the players (elements). Can be strings (names), integers (indicies), and tuples.

        objective_function (Callable):
            The game (in-silico experiment). It should get the complement set and return one numeric value
            either int or float.
            This function is just calling it as: objective_function(complement, **objective_function_params)

            An example using networkx with some tips:

            def lesion_me_senpai(complements, network, index):
                # note "index", your function should be able to track the effects on the target and the keyword for
                  that is "index"

                if len(complements) == len(A)-1:  # -1 since the target node is active
                    return 0

                lesioned_network = deepcopy(network)
                for target in complements:
                    lesioned_network[target] = 0  # setting all connections of the targets to 0

                activity = network.run(lesioned_network) # or really, whatever you want!
                return float(activity[index].mean())

            (you sometimes need to specify what should happen during edge-cases like an all-lesioned network)


        objective_function_params (Optional[Dict]):
            Kwargs for the objective_function. A dictionary pair of {'index': index} will be added to this during
            the process so your function can track the lesion effect.

        multiprocessing_method (str = 'joblib'):
            So far, two methods of parallelization is implemented, 'joblib' and 'ray' and the default method is joblib.
            If using ray tho, you need to decorate your objective function with @ray.remote decorator. Visit their
            documentations to see how to go for it.

        n_parallel_games (int = -1):
            Number of parallel games. Default is -1, which means all cores so it can make the system
            freeze for a short period, if that happened then maybe go for -2, which means one msapy is
            left out. Or really just specify the number of threads you want to use!

        n_permutations (int = 1000):
            Number of permutations per node.
            Didn't check it systematically yet but just based on random explorations
            I'd say something around 1000 is enough.

        permutation_seed (Optional[int] = None):
            Sets the random seed of the sampling process. Default is None so if nothing is given every call results in
            a different orderings.

        index : index to be passed to the objective function

        element : element whose causal influence we want to calculate.

    Returns:
        causal_influences (pd.DataFrame)
        multi_scores (bool)
        is_timeseries (bool)

    """

    objective_function_params['index'] = index

    # Takes the target out of the to_be_lesioned list
    without_target = set(elements).difference({element})

    shapley_output = interface(n_permutations=n_permutations,
                               elements=list(without_target),
                               dual_progress_bars=False,
                               objective_function=objective_function,
                               objective_function_params=objective_function_params,
                               n_parallel_games=n_parallel_games,
                               multiprocessing_method=multiprocessing_method,
                               random_seed=permutation_seed,
                               lazy=lazy,
                               save_permutations=False,
                               mbar=mbar)

    if shapley_output.contribution_type == "scaler":
        return shapley_output.shapley_values, shapley_output.contribution_type
    return shapley_output, shapley_output.contribution_type
