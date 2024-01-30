import numpy as np
import pytest
from msapy import msa, utils as ut


# ------------------------------#
# A function that assigns 1 to the cause and 0 to others
def simple(complements, causes):
    if len(causes) != 0 and {causes}.issubset(complements):
        return 0
    else:
        return 1


def simple_with_interaction(complements):
    if ("A115" not in complements) and ("B655" not in complements):
        return sum(contrib_dict.values()) - sum(contrib_dict[k] for k in complements) + 87

    return sum(contrib_dict.values()) - sum(contrib_dict[k] for k in complements)


# ------------------------------#
elements = ['A115', 'b', 'c']
cause = 'A115'
shapley_table = msa.interface(
    elements=elements,
    n_permutations=300,
    objective_function=simple,
    n_parallel_games=1,
    objective_function_params={'causes': cause},
    random_seed=111)

contrib_dict = {"A115": 10, "B655": 9, "C": 57, "D": -8, "E": 42}

# ------------------------------#


def test_max():
    assert shapley_table.mean().max() == 1


def test_min():
    assert shapley_table.mean().min() == 0


def test_cause():
    assert shapley_table['A115'].mean() == 1


def test_others():
    assert shapley_table['b'].mean() == 0
    assert shapley_table['c'].mean() == 0


def test_d_index():
    assert ut.distribution_of_processing(
        shapley_vector=shapley_table.mean()) == 0


@pytest.mark.parametrize("n_parallel_games, multiprocessing_method, lazy", [(1, 'joblib', True), (-1, 'joblib', True), (1, 'joblib', False), (-1, 'joblib', False)])
def test_interaction_2d(n_parallel_games, multiprocessing_method, lazy):
    interactions = msa.network_interaction_2d(
        elements=list(contrib_dict.keys()),
        n_permutations=1000,
        objective_function=simple_with_interaction,
        n_parallel_games=n_parallel_games,
        multiprocessing_method=multiprocessing_method,
        random_seed=111,
        lazy=lazy)

    expected_interactions = np.array([[0, 87, 0, 0, 0], [87, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    assert np.allclose(expected_interactions, interactions, atol=1e-3)


@pytest.mark.parametrize("n_cores, multiprocessing_method, parallelize_over_games",
                         [(1, 'joblib', True), (-1, 'joblib', True), (1, 'joblib', False), (-1, 'joblib', False)])
def test_estimate_causal_influence(n_cores, multiprocessing_method, parallelize_over_games):
    true_causal_influences = np.random.normal(0, 5, (4, 4))
    np.fill_diagonal(true_causal_influences, 0)

    def objective_function_causal_influence(complements, index):
        return true_causal_influences[index].sum() - true_causal_influences[index, complements].sum()
    calculated_causal_influences = msa.estimate_causal_influences(elements=list(range(4)),
                                                                  n_permutations=1000,
                                                                  objective_function=objective_function_causal_influence,
                                                                  multiprocessing_method=multiprocessing_method,
                                                                  parallelize_over_games=parallelize_over_games,
                                                                  n_cores=n_cores).values

    np.fill_diagonal(calculated_causal_influences, 0)

    assert np.allclose(calculated_causal_influences, true_causal_influences)


def test_interface_parallel_joblib():
    shapley_table_parallel = msa.interface(
        elements=elements,
        n_permutations=300,
        objective_function=simple,
        n_parallel_games=-1,
        objective_function_params={'causes': cause},
        random_seed=111)

    assert shapley_table.equals(shapley_table_parallel)
