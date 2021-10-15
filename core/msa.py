import pandas as pd
import random
from typing import Callable, Optional, Dict
from typeguard import typechecked
import utils as ut
from ordered_set import OrderedSet


@typechecked
def make_permutation_space(*,
                           elements: list,
                           n_permutations: int) -> list:
    permutation_space = []
    for sample in range(n_permutations):
        permutation_space.append(tuple(random.sample(elements, len(elements))))
    return permutation_space


@typechecked
def make_combination_space(*, permutation_space: list) -> OrderedSet:
    combination_space = OrderedSet()
    for permutation in permutation_space:
        for index, element in enumerate(permutation):

            including = frozenset(permutation[:index + 1])
            excluding = frozenset(permutation[:index])

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
