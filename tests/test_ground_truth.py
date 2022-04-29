import numpy as np
from msapy import msa, utils as ut


# ------------------------------#
# A function that assigns 1 to the cause and 0 to others
def simple(complements, causes):
    if len(causes) != 0 and set(causes).issubset(complements):
        return 0
    else:
        return 1


def simple_with_interaction(complements):
    if ("A" not in complements) and ("B" not in complements):
        return sum(contrib_dict.values()) - sum(contrib_dict[k] for k in complements) + 87

    return sum(contrib_dict.values()) - sum(contrib_dict[k] for k in complements)


# ------------------------------#
elements = ['a', 'b', 'c']
cause = 'a'
shapley_table, contributions, lesions = msa.interface(
    elements=elements,
    n_permutations=300,
    objective_function=simple,
    n_parallel_games=1,
    objective_function_params={'causes': cause},
    random_seed=111)

contrib_dict = {"A": 10, "B": 9, "C": 57, "D": -8, "E": 42}

# ------------------------------#


def test_max():
    assert shapley_table.mean().max() == 1


def test_min():
    assert shapley_table.mean().min() == 0


def test_cause():
    assert shapley_table['a'].mean() == 1


def test_others():
    assert shapley_table['b'].mean() == 0
    assert shapley_table['c'].mean() == 0


def test_num_combinations():
    assert len(contributions) == 2 ** 3


def test_d_index():
    assert ut.distribution_of_processing(
        shapley_vector=shapley_table.mean()) == 0


def test_interaction_2d():
    interactions = msa.network_interaction_2d(
        elements=list(contrib_dict.keys()),
        n_permutations=1000,
        objective_function=simple_with_interaction,
        n_parallel_games=1)
    expected_interactions = np.array([[0, 87, 0, 0, 0], [87, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    assert np.allclose(expected_interactions, interactions)
