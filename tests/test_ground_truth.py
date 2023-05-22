import numpy as np
import pytest
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
shapley_table = msa.interface(
    elements=elements,
    n_permutations=300,
    objective_function=simple,
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


def test_d_index():
    assert ut.distribution_of_processing(
        shapley_vector=shapley_table.mean()) == 0


def test_interaction_2d():
    interactions = msa.network_interaction_2d(
        elements=list(contrib_dict.keys()),
        n_permutations=1000,
        objective_function=simple_with_interaction,
        random_seed=111)

    expected_interactions = np.array([[0, 87, 0, 0, 0], [87, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    assert np.allclose(expected_interactions, interactions, atol=1e-3)


def test_estimate_causal_influence():
    true_causal_influences = np.random.normal(0, 5, (4, 4))
    np.fill_diagonal(true_causal_influences, 0)

    def objective_function_causal_influence(complements, index):
        return true_causal_influences[index].sum() - true_causal_influences[index, complements].sum()
    calculated_causal_influences = msa.estimate_causal_influences(elements=list(range(4)),
                                                                  n_permutations=1000,
                                                                  objective_function=objective_function_causal_influence).values

    np.fill_diagonal(calculated_causal_influences, 0)

    assert np.allclose(calculated_causal_influences, true_causal_influences)
