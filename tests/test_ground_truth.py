import math
from msapy import msa, utils as ut


# ------------------------------#
# A function that assigns 1 to the cause and 0 to others
def simple(complements, causes):
    if len(causes) != 0 and set(causes).issubset(complements):
        return 0
    else:
        return 1


def simple_with_interaction(complements, lesioned_element=None, paired_element=None):
    if lesioned_element:
        complements = (*complements, lesioned_element)

    if ("A" not in complements) and ("B" not in complements):
        return sum(contrib_dict.values()) - sum(contrib_dict[k] for k in complements) + contrib_dict["A"] + contrib_dict["B"]

    if paired_element:
        complements = (*complements, paired_element)

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
gamma_AB, gamma_A, gamma_B = msa.interface_2d(
    elements=list(contrib_dict.keys()),
    pair=("A", "B"),
    n_permutations=5000,
    objective_function=simple_with_interaction,
    n_parallel_games=1)

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


def test_interface_2d():
    assert math.isclose(gamma_AB, 38)
    assert math.isclose(gamma_A, 10)
    assert math.isclose(gamma_B, 9)
