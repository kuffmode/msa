from msapy import msa, utils as ut
import pytest


@pytest.mark.parametrize("n_permutations", [-1, 0, -100])
def test_make_permutation_space_n_permutation_value_error(n_permutations):
    elements = ['a', 'b', 'c']
    with pytest.raises(ValueError):
        msa.make_permutation_space(elements=elements,
                                   n_permutations=n_permutations)


@pytest.mark.parametrize("n_permutations", [-3.0, 5.0, 1e3, "4"])
def test_make_permutation_space_n_permutation_type_error(n_permutations):
    elements = ['a', 'b', 'c']
    with pytest.raises(TypeError):
        msa.make_permutation_space(elements=elements,
                                   n_permutations=n_permutations)


@pytest.mark.parametrize("elements", [[8, 5], [5, 'g', 8], [4, 4.5, 5]])
def test_make_permutation_space_elements_value_error(elements):
    n_permutations = 10
    with pytest.raises(ValueError):
        msa.make_permutation_space(elements=elements,
                                   n_permutations=n_permutations)


@pytest.mark.parametrize("elements", [(5, 6, 8), {8, 5, 9}, range(5)])
def test_make_permutation_space_elements_type_error(elements):
    n_permutations = 10
    with pytest.raises(TypeError):
        msa.make_permutation_space(elements=elements,
                                   n_permutations=n_permutations)


@pytest.mark.parametrize("permutation_space", [["a", "bb", "c"], [[1, 2, 4], [4, 1], [4, 2, 1]]])
def test_make_combination_space_permutation_space_value_error(permutation_space):
    with pytest.raises(ValueError):
        msa.make_combination_space(permutation_space=permutation_space)


@pytest.mark.parametrize("p_value", [0, -1])
def test_zero_negative_pvalue(p_value):
    elements = ['a', 'b', 'c']

    def one(complements):
        return 1

    for_bootstrap = msa.interface(elements=elements,
                                  n_permutations=100,
                                  objective_function=one)
    with pytest.raises(ValueError):
        ut.bootstrap_hypothesis_testing(dataset=for_bootstrap,
                                        p_value=p_value)
