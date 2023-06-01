import warnings
import numpy as np

from typeguard import typechecked
from itertools import groupby
from numbers import Number
from typing import Tuple, Union


def _check_valid_permutation_space(permutation_space: list):
    if not all(_is_iterable(permutation) for permutation in permutation_space):
        raise ValueError(
            "Found at least one permutation in permutation space that is not iterable.")

    if not _are_lists_same_length(permutation_space):
        raise ValueError("All permutations should be of the same length.")


def _check_valid_elements(elements: list):
    if len(elements) < 3:
        raise ValueError(
            "The number of elements should be at least 3 to perform MSA")

    if not _is_homogeneous_list(elements):
        raise ValueError('All elements should be of the same type.'
                         ' At least two elements of different types found')

    if not _is_sortable(elements[0]):
        raise ValueError('The elements need to be comparable to each other')


def _check_valid_n_permutations(n_permutations):
    if n_permutations <= 0:
        raise ValueError(
            "Specified number of permutations doesn't make sense because it's either zero or smaller.")
    elif 1 < n_permutations < 100:
        warnings.warn("Specified number of permutations is kinda small so the results might not be as accurate.",
                      stacklevel=2)


def _check_valid_combination_space(combination_space, elements, lesioned):
    diff = max(combination_space, key=len) ^ elements - \
        {lesioned if lesioned else None}

    # ------------------------------#
    if len(diff) != 0:
        raise ValueError(f"Elements in the combination space are different from what's in the elements list."
                         f"The symmetric difference-set is: {list(diff)}")
    # ------------------------------#


@typechecked
def _get_contribution_type(contributions: dict) -> Tuple[str, Union[dict, np.number, Number, np.ndarray]]:

    if not _is_homogeneous_list(list(contributions.values())):
        raise ValueError(
            "Objective function returned values of different data types")

    arbitrary_contribution = next(iter(contributions.values()))

    if isinstance(arbitrary_contribution, np.ndarray):
        return "nd", arbitrary_contribution

    if not _is_number(arbitrary_contribution):
        raise ValueError("Objective function should return a value that is either"
                         " a Number, a dictionary or Numbers, iterable of numbers, or a numpy array."
                         f" Returned {type(arbitrary_contribution)} instead.")

    return "scaler", arbitrary_contribution


def _check_get_shapley_table_args(contributions, objective_function, lazy):
    if lazy == True:
        if objective_function is None:
            raise ValueError(
                "Objective function should be passed in case of lazy calculation of shapely table")
        if contributions is not None:
            raise ValueError(
                "A contributions dictionacontributions_excludingry is not required in case of lazy calculation of shapely table")
    else:
        if contributions is None:
            raise ValueError(
                "A contributions dictionary is neccessary for the calculation of shapley table if lazy is set to False")


def _is_number(x) -> bool:
    return isinstance(x, (Number, np.number))


@typechecked
def _are_lists_same_length(lists: list):
    return len({len(i) for i in lists}) == 1


@typechecked
def _is_homogeneous_list(l: list) -> bool:
    return len(list(groupby(l, type))) == 1


@typechecked
def _is_iterable(obj: object) -> bool:
    """Checks if the object passed is an iterable. Uses ducktyping.

    Args:
        obj (object): 

    Returns:
        bool: returns True if object is iterable
    """
    try:
        iter(obj)
    except Exception:
        return False
    else:
        return True


def _is_sortable(obj):
    cls = obj.__class__
    return cls.__lt__ != object.__lt__ or cls.__gt__ != object.__gt__
