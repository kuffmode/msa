import warnings
from typing import Callable, Optional, Dict, Tuple, Union
import numpy as np
import pandas as pd
from ordered_set import OrderedSet
from itertools import combinations
from typeguard import typechecked

import dask
from dask.delayed import Delayed
from dask.diagnostics import ProgressBar

from msapy.checks import _check_valid_combination_space, _check_valid_elements, _check_valid_n_permutations, _check_valid_permutation_space, _get_contribution_type, _is_number
from msapy.datastructures import ShapleyModeND, ShapleyTable


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
    lesioned = set(lesioned) if lesioned else set()

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
def get_shapley_table(*,
                      permutation_space: list,
                      objective_function: Optional[Callable],
                      lesioned: Optional[any] = None,
                      objective_function_params: Optional[Dict] = None) -> Delayed:
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

    Returns:
        pd.DataFrame: Shapley table or a dict of Shapely tables, columns will be 
        elements and indices will be samples (permutations). 
        It will be a Multi-Index DataFrame if the contributions are a dict
        i.e. the objective function returns multiple score functions (eg. accuracy, f1_score, etc.)
        It will be a Multi-Index DataFrame if the contributions are a timeseries.
        The index at `level=1` will be the timestamps
    """
    _check_valid_permutation_space(permutation_space)

    contribution_type, arbitrary_contrib = _get_contribution_type(objective_function(tuple(), **objective_function_params))

    lesioned = set(lesioned) if lesioned else set()
    shapley_table = {} if contribution_type != 'nd' else 0
        
    for i, permutation in enumerate(set(permutation_space)):
        isolated_contributions = []  # got to be a better way!
        # iterate over all elements in the permutation to calculate their isolated contributions
        for index, _ in enumerate(permutation):
            including = frozenset(permutation[:index + 1]) - lesioned
            excluding = frozenset(permutation[:index]) - lesioned

            # the isolated contribution of an element is the difference of contribution with that element and without that element
            contributions_including = dask.delayed(objective_function)(tuple(excluding), **objective_function_params)
            contributions_excluding = dask.delayed(objective_function)(tuple(including), **objective_function_params)

            isolated_contributions.append(contributions_including - contributions_excluding)
            
        if contribution_type == 'nd':
            isolated_contributions = [x for _, x in sorted(zip(permutation, isolated_contributions))]
            shapley_table = shapley_mean(shapley_table, i, isolated_contributions)
        else:
            shapley_table[permutation] = np.array(isolated_contributions)

    # post processing of shapley values based on what type of contribution it is. The format of output will vary based on if the
    # values are multi-scores, timeseries, etc.
    if contribution_type == 'nd':
        shapley_table = shapley_table.reshape(shapley_table.shape[0], -1).T
        shapley_table = dask.delayed(pd.DataFrame)(shapley_table, columns=sorted(permutation))
        return dask.delayed(ShapleyModeND)(shapley_table, arbitrary_contrib.shape)

    shapley_table = dask.delayed(pd.DataFrame)([dict(zip(permutations, shapleys))
                                    for permutations, shapleys in shapley_table.items()])
    return dask.delayed(ShapleyTable)(shapley_table)


@dask.delayed
def shapley_mean(shapley_table, i, isolated_contributions):
    return shapley_table + (np.array(isolated_contributions) -
                              shapley_table) / (i + 1)

@dask.delayed
def temp_name(shapley_table, i):
    return [dict(zip(permutations, shapleys[:, i]))
                                            for permutations, shapleys in shapley_table.items()]


@typechecked
def interface(*,
              n_permutations: int,
              elements: list,
              objective_function: Callable,
              objective_function_params: Dict = {},
              permutation_space: Optional[list] = None,
              pair: Optional[Tuple] = None,
              lesioned: Optional[any] = None,
              rng: Optional[np.random.Generator] = None,
              random_seed: Optional[int] = None,
              return_delayed: bool = False,
              ) -> Union[Delayed, pd.DataFrame]:
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
            sets the random seed of tdual_progress_bars=dual_progress_bars,he sampling process. Only used when `rng` is None. Default is None.

        n_parallel_games (int):
            Number of parallel jobs (number of to-be-occupied cores),
            -1 means all CPU cores and 1 means a serial process.
            I suggest using 1 for debugging since things get messy in parallel!

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

    shapley_table = get_shapley_table(permutation_space=permutation_space,
                                        lesioned=lesioned,
                                        objective_function=objective_function,
                                        objective_function_params=objective_function_params)[elements]
    if return_delayed:
        return shapley_table
    
    with ProgressBar():
        return shapley_table.compute()


@typechecked
def interaction_2d(*,
                   n_permutations: int,
                   elements: list,
                   pair: tuple,
                   objective_function: Callable,
                   objective_function_params: Dict = {},
                   rng: Optional[np.random.Generator] = None,
                   random_seed: Optional[int] = None,
                   return_delayed: bool = False
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
                      "rng": rng,
                      "random_seed": random_seed,
                      "return_delayed": True}

    # calculate the shapley values with element j lesioned
    shapley_i = interface(**interface_args, lesioned=pair[1])
    # get the shapley value of element i with element j leasioned
    gamma_i = _get_gamma(shapley_i, pair[0]).sum()

    # calculate the shapley values with element i lesioned
    shapley_j= interface(**interface_args, lesioned=pair[0])
    # get the shapley value of element j with element i leasioned
    gamma_j = _get_gamma(shapley_j, pair[1]).sum()

    # calculate the shapley values with element i and j together in every combination
    shapley_ij = interface(**interface_args, pair=pair)
    # get the sum of the shapley value of element i and j
    gamma_ij = _get_gamma(shapley_ij, pair).sum()

    if return_delayed:
        return gamma_ij, gamma_i, gamma_j
    
    with ProgressBar():
        return dask.compute(gamma_ij, gamma_i, gamma_j)


@typechecked
def network_interaction_2d(*,
                           n_permutations: int,
                           elements: list,
                           pairs: Optional[list] = None,
                           objective_function: Callable,
                           objective_function_params: Dict = {},
                           rng: Optional[np.random.Generator] = None,
                           random_seed: Optional[int] = None,
                           return_delayed: bool = False
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
                      "rng": rng,
                      "random_seed": random_seed,
                      "return_delayed": True}

    interactions = np.zeros((len(elements), len(elements)))

    # iterate over all the pairs to run interaction_2d
    for x, y in all_pairs:
        gammaAB, gammaA, gammaB = interaction_2d(pair=(elements[x], elements[y]),
                                                 **interface_args)
        # if not _is_number(gammaAB):
        #     raise NotImplementedError("`network_interaction_2d` does not work with"
        #                               " timeseries or multiscore contributions yet.")
        interactions[x, y] = interactions[y, x] = gammaAB - gammaA - gammaB
    if return_delayed:
        return interactions
    
    with ProgressBar():
        return dask.compute(interactions)


@dask.delayed
def _get_gamma(shapley_table, idx):
    """returns shapley value of elements in idx

    Args:
        shapley_table (pd.DataFrame): shapley table with one element lesioned
        idx (_type_): element of interest

    Returns:
        shapley value of elements in idx
    """
    if isinstance(shapley_table, ShapleyTable):
        gamma = shapley_table.shapley_values[list(idx)]
    else:
        gamma = ShapleyModeND[list(idx)]
    return gamma


@typechecked
def estimate_causal_influences(elements: list,
                               objective_function: Callable,
                               objective_function_params: Optional[dict] = None,
                               target_elements: Optional[list] = None,
                               n_permutations: int = 1000,
                               permutation_seed: Optional[int] = None,
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

    results = [causal_influence_single_element(elements, objective_function,
                                                objective_function_params, n_permutations,
                                                permutation_seed, index, element, return_delayed=True) for index, element in enumerate(target_elements)]
    
    with ProgressBar():
        results = dask.compute(results)[0]

    _, contribution_type = results[0]

    if contribution_type=="scaler":
        shapley_values = [r[0].shapley_values for r in results]
        return pd.DataFrame(shapley_values, columns=elements)


    shapley_values = [r[0] for r in results]
    causal_influences = pd.concat(shapley_values, keys=elements)
    
    return causal_influences[causal_influences.index.levels[0]]


def causal_influence_single_element(elements, objective_function,
                                    objective_function_params, n_permutations,
                                    permutation_seed, index, element,
                                    return_delayed: bool = False):
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
                                                objective_function=objective_function,
                                                objective_function_params=objective_function_params,
                                                random_seed=permutation_seed,
                                                return_delayed=True)
    
    contribution_type, _ = _get_contribution_type(objective_function(tuple(), **objective_function_params))

    if return_delayed:
        return shapley_output, contribution_type
    
    with ProgressBar():
        return shapley_output.compute(), contribution_type