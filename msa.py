import random
from typing import Callable, Optional, Dict

import pandas as pd
from ordered_set import OrderedSet
from typeguard import typechecked

import utils as ut


@typechecked
def make_permutation_space(*,
                           elements: list,
                           n_permutations: int) -> list:
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

    Returns (list[tuple]):
        Permutation space as a list of lists with shape (n_permutations, len(elements))
    """
    permutation_space = []  # TODO: How not to make an empty placeholder first!
    for sample in range(n_permutations):
        permutation_space.append(tuple(random.sample(elements, len(elements))))
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

            including = frozenset(permutation[:index + 1])  # forming the coaliting with the target element
            excluding = frozenset(permutation[:index])  # forming it without the target element

            # It's possible to end up with the same coalitions many times so:
            if including not in combination_space:
                combination_space.add(including)

            if excluding not in combination_space:
                combination_space.add(excluding)

    return combination_space


@typechecked
def take_contributions(*,
                       elements: list,
                       combination_space: OrderedSet,
                       objective_function: Callable,
                       objective_function_params: Optional[Dict] = None) -> Dict:
    """
    This function fills up the combination_space with the game you define (objective function). There is an important
    point to keep in mind, Shapley values are the added contributions of elements while in MSA we calculate them by
    perturbation so although it's intuitive to think the combination in combination space is the element that will be
    lesioned, it is not the case, it will be everything else but the coalition, i.e., the target coalition are the
    only intact elements. This function takes care of this by passing the complement of each coalition to the
    game while assigning the results to the target coalition, just keep the logic in mind.

    A second point is that this function returns a filled combination_space, it is not filling it in-place for the
    sake of purity.

    Args:
        elements (list):
            List of the players. Obviously, should be the same passed to make_permutation.

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
    objective_function_params = objective_function_params if objective_function_params else {}

    for combination in ut.generatorize(to_iterate=combination_space):
        complement = tuple(elements.difference(combination))  # lesion everything but the target coalition
        contributions[combination] = objective_function(complement, **objective_function_params)

    return contributions


@typechecked
def make_shapley_values(*,
                        contributions: Dict,
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
        temp = []  # got to be a better way!

        # if the set is small it's possible that the permutation space exhausts the combination space so:
        if permutation not in shapley_table:
            for index, element in enumerate(ut.generatorize(to_iterate=permutation)):
                including = frozenset(permutation[:index + 1])
                excluding = frozenset(permutation[:index])
                temp.append(contributions[including] - contributions[excluding])
            shapley_table[permutation] = temp

    shapley_values = pd.DataFrame([dict(zip(perms, vals)) for perms, vals in shapley_table.items()])
    return shapley_values
