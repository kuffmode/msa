import random
import warnings
from typing import Callable, Optional, Dict, Tuple, Union
import numpy as np
import pandas as pd
from ordered_set import OrderedSet
from typeguard import typechecked

from msapy import utils as ut


@typechecked
def make_permutation_space(*,
                           elements: list,
                           n_permutations: int,
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

        random_seed (Optional[int]):
            sets the random seed of the sampling process. Default is None so if nothing is given every call results in
            a different orderings.

    Returns (list[tuple]):
        Permutation space as a list of lists with shape (n_permutations, len(elements))
    """

    # ------------------------------#
    if n_permutations <= 0:
        raise ValueError("Specified number of permutations doesn't make sense because it's either zero or smaller.")
    elif 1 < n_permutations < 100:
        warnings.warn("Specified number of permutations is kinda small so the results might not be as accurate.",
                      stacklevel=2)
    # ------------------------------#
    if random_seed:
        random.seed(random_seed)

    permutation_space = [tuple(random.sample(elements, len(elements))) for _ in range(n_permutations)]
    return permutation_space


@typechecked
def make_combination_space(*, permutation_space: list) -> OrderedSet:
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

    Returns (OrderedSet):
        Combination space as an OrderedSet of frozensets.
    """

    combination_space = OrderedSet()
    for permutation in ut.generatorize(to_iterate=permutation_space):

        for index, _ in enumerate(permutation):  # we really don't care about the element itself here

            including = frozenset(permutation[:index + 1])  # forming the coalition with the target element
            excluding = frozenset(permutation[:index])  # forming it without the target element

            # It's possible to end up with the same coalitions many times so:
            if including not in combination_space:
                combination_space.add(including)

            if excluding not in combination_space:
                combination_space.add(excluding)

    return combination_space


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
    diff = combination_space.items[len(elements)] ^ elements

    # ------------------------------#
    if len(diff) != 0:
        raise ValueError(f"Elements in the combination space are different from what's in the elements list."
                         f"The symmetric difference-set is: {list(diff)}")
    # ------------------------------#

    complement_space = OrderedSet()
    for combination in ut.generatorize(to_iterate=combination_space):
        complement_space.add(tuple(elements.difference(combination)))
    return complement_space


@typechecked
def take_contributions(*,
                       elements: list,
                       complement_space: OrderedSet,
                       combination_space: OrderedSet,
                       objective_function: Callable,
                       objective_function_params: Optional[Dict] = None) -> Tuple[Dict, Dict]:
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

    Returns (Dict):
        A dictionary of combinations:results
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

    for combination in ut.generatorize(to_iterate=combination_space):
        complement = tuple(elements.difference(combination))  # lesion everything but the target coalition
        result = objective_function(complement, **objective_function_params)

        contributions[combination] = result
        lesion_effects[complement] = result
    return contributions, lesion_effects


@typechecked
def make_shapley_values(*,
                        contributions: Union[Dict, Tuple[Dict, Dict]],
                        permutation_space: list) -> pd.DataFrame:
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

    Returns (pd.DataFrame):
        Shapley table, columns will be elements and indices will be samples (permutations).
    """
    shapley_table = {}
    for permutation in ut.generatorize(to_iterate=permutation_space):
        isolated_contributions = []  # got to be a better way!

        # if the set is small it's possible that the permutation space exhausts the combination space so:
        if permutation not in shapley_table:
            for index, element in enumerate(ut.generatorize(to_iterate=permutation)):
                including = frozenset(permutation[:index + 1])
                excluding = frozenset(permutation[:index])

                isolated_contributions.append(contributions[including] - contributions[excluding])
            shapley_table[permutation] = isolated_contributions

    shapley_values = pd.DataFrame([
        dict(zip(permutations, shapleys)) for permutations, shapleys in shapley_table.items()])
    return shapley_values


@typechecked
def interface(*,
              n_permutations: int,
              n_parallel_games: int = -1,
              elements: list,
              objective_function: Callable,
              objective_function_params: Optional[Dict] = None,
              permutation_space: Optional[list] = None,
              multiprocessing_method: str = 'joblib',
              random_seed: Optional[int] = None,
              ) -> Tuple[pd.DataFrame, Dict, Dict]:
    """
    A wrapper function to call other related functions internally and produces an easy-to-use pipeline.

    Args:
        n_permutations (int):
            Number of permutations (samples) per element.

        n_parallel_games (int):
            Number of parallel jobs (number of to-be-occupied cores),
            -1 means all CPU cores and 1 means a serial process.
            I suggest using 1 for debugging since things gets messy in parallel!

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

        objective_function_params (Optional[Dict]):
           Kwargs for the objective_function.

        permutation_space (Optional[list]):
            Already generated permutation space, in case you want to be more reproducible or something and use the same
            lesion combinations for many metrics.

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

        random_seed (Optional[int]):
            sets the random seed of the sampling process. Default is None so if nothing is given every call results in
            a different orderings.

    Returns ([pd.DataFrame, Dict, Dict]):
        shapley_table, contributions, lesion_effects

        Note that contributions and lesion_effects are the same values, addressed differently. For example:
        If from a set of ABCD removing AC ends with some value x, you can say the contribution of BD=x and the
        effect of removing AC=x. So the same values are addressed differently in the two returned Dicts.
        Of course, it makes more sense to compare the lesion effects with the intact system but who am I to judge.
    """
    of_params = objective_function_params if objective_function_params else {}

    if not permutation_space:
        permutation_space = make_permutation_space(elements=elements,
                                                   n_permutations=n_permutations,
                                                   random_seed=random_seed)
    else:
        warnings.warn("A Permutation space is given so n_permutations will fall back to what's specified there.",
                      stacklevel=2)

    combination_space = make_combination_space(permutation_space=permutation_space)
    complement_space = make_complement_space(combination_space=combination_space,
                                             elements=elements)

    if n_parallel_games == 1:
        contributions, lesion_effects = take_contributions(elements=elements,
                                                           complement_space=complement_space,
                                                           combination_space=combination_space,
                                                           objective_function=objective_function,
                                                           objective_function_params=of_params)
    else:
        contributions, lesion_effects = ut.parallelized_take_contributions(
            multiprocessing_method=multiprocessing_method,
            n_cores=n_parallel_games,
            complement_space=complement_space,
            combination_space=combination_space,
            objective_function=objective_function,
            objective_function_params=of_params)

    shapley_values = make_shapley_values(contributions=contributions, permutation_space=permutation_space)
    return shapley_values, contributions, lesion_effects


@typechecked
def estimate_causal_influences(network_connectome: np.ndarray,
                               objective_function: Callable,
                               objective_function_params: Optional[Dict],
                               multiprocessing_method: str = 'joblib',
                               n_cores: int = -1,
                               n_permutations: int = 1000,
                               permutation_seed: Optional[int] = None,
                               ) -> pd.DataFrame:
    """
    Estimates the causal contribution (Shapley values) of each node on the rest of the network. Basically, this function
    performs MSA iteratively on each node and tracks the changes in the objective_function of the taret node.
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
        your specific case but based on experience, it's not a good idea.

        and 5. Shapley values sum up (or will be close) to the value of the intact coalition. So for example if the
        mean activity of node C here is 50 then causal_contribution_matrix.sum(axis=0) = 50 or close to 50. If not it
        means:
            1. the number of permutations are not enough
            2. there is randomness somewhere in the process
            3. your objective function is not suitable


    Args:
        network_connectome (np.ndarray):
            Structural backbone of your network.
            I probably can turn this into a list of elements instead so #TODO.

        objective_function (Callable):
            The game (in-silico experiment). It should get the complement set and return one numeric value
            either int or float.
            This function is just calling it as: objective_function(complement, **objective_function_params)

            An example using networkx with some tips:
            (you sometimes need to specify what should happen during edge-cases like an all-lesioned network)

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

        objective_function_params (Optional[Dict]):
            Kwargs for the objective_function. A dictionary pair of {'index': index} will be added to this during
            the process so your function can track the lesion effect.

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

    Returns:

    """

    # Initialize the stuff
    nodes = [node for node,_ in enumerate(network_connectome)]
    permutations = dict()
    combinations = dict()
    complements = dict()
    contributions = dict()
    lesion_effects = dict()
    shapley_values = dict()
    objective_function_params = objective_function_params if objective_function_params else {}

    # Looping through the nodes
    for index, _ in enumerate(network_connectome):
        print(f"working on the node number {index} from {len(network_connectome)} nodes.")
        objective_function_params['index'] = index

        # Takes the target out of the to_be_lesioned list
        without_target = set(nodes).difference({index})

        # Generates N permutations and their respective combinations/complements of the network without the target.
        permutations[index] = make_permutation_space(n_permutations=n_permutations,
                                                     elements=list(without_target),
                                                     random_seed=permutation_seed)
        combinations[index] = make_combination_space(permutation_space=permutations[index])
        complements[index] = make_complement_space(combination_space=combinations[index],
                                                   elements=list(without_target))

        # Plays the game for each lesion combination
        contributions[index], lesion_effects[index] = ut.parallelized_take_contributions(
            multiprocessing_method=multiprocessing_method,
            n_cores=n_cores,
            complement_space=complements[index],
            combination_space=combinations[index],
            objective_function=objective_function,
            objective_function_params=objective_function_params)

        # Calculates the good-old Shapley values for the source nodes
        shapley_values[index] = make_shapley_values(contributions=contributions[index],
                                                    permutation_space=permutations[index])

    # The value orders are not sorted so
    for shapley_value in shapley_values:
        shapley_values[shapley_value] = shapley_values[shapley_value].sort_index(axis=1).mean(axis=0)
    causal_influences = pd.DataFrame.from_dict(shapley_values)
    return causal_influences
