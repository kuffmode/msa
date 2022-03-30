from cmath import exp
from msapy import msa, utils as ut
import pytest


# ------------------------------#
# A function that assigns 1 to the cause and 0 to others
def simple(complements):
    contribution = {"score_1": 1, "score_2": 1}
    if 'a' in complements:
        contribution['score_1'] = 0
    if 'b' in complements:
        contribution['score_2'] = 0

    return contribution


# ------------------------------#
elements = ['a', 'b', 'c']
shapley_table, contributions, lesions = msa.interface(
    elements=elements,
    n_permutations=300,
    objective_function=simple,
    n_parallel_games=1,
    random_seed=111)
# ------------------------------#

expected_contributions = {("score_1", "a"): 1, ("score_2", "a"): 0,
                          ("score_1", "b"): 0, ("score_2", "b"): 1,
                          ("score_1", "c"): 0, ("score_2", "c"): 0}


@pytest.mark.parametrize("score, element", list(expected_contributions.keys()))
def test_contributions(score, element):
    assert shapley_table.loc[score][element].mean(
    ) == expected_contributions[(score, element)]
