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


@typechecked
def _get_contribution_type(contributions: dict) -> Tuple[Union[dict, float, np.ndarray], bool, bool]:

    if not _is_homogeneous_list(list(contributions.values())):
        raise ValueError(
            "Objective function returned values of different data types")

    arbitrary_contribution = next(iter(contributions.values()))
    multi_scores = isinstance(arbitrary_contribution, dict)
    is_timeseries = _is_iterable(arbitrary_contribution) and (not multi_scores)

    if is_timeseries and not _is_number(arbitrary_contribution[0]):
        raise ValueError("Objective function should return a value that is either"
                         " a Number, a dictionary or Numbers, or an iterable of numbers")

    if not (multi_scores or is_timeseries or _is_number(arbitrary_contribution)):
        raise ValueError("Objective function should return a value that is either"
                         " a Number, a dictionary or Numbers, or an iterable of numbers")

    return arbitrary_contribution, multi_scores, is_timeseries


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
