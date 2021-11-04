from msapy import msa, utils as ut
import pytest
import ray


def test_negative_integers():
    elements = ['a', 'b', 'c']

    def one(complements):
        return 1

    with pytest.raises(ValueError):
        msa.make_permutation_space(elements=elements,
                                   n_permutations=-1)

        msa.interface(elements=elements,
                      n_permutations=-1,
                      objective_function=one,
                      )


def test_zero_permutations():
    elements = ['a', 'b', 'c']

    def one(complements):
        return 1

    with pytest.raises(ValueError):
        msa.make_permutation_space(elements=elements,
                                   n_permutations=0)

        msa.interface(elements=elements,
                      n_permutations=0,
                      objective_function=one,
                      )


def test_confusing_parallelstuff():
    elements = ['a', 'b', 'c']

    def one(complements):
        return 1

    @ray.remote
    def two(complements):
        return 2

    with pytest.raises(ValueError):
        msa.interface(elements=elements,
                      n_permutations=0,
                      objective_function=one,
                      multiprocessing_method='ray'
                      )

        msa.interface(elements=elements,
                      n_permutations=0,
                      objective_function=two,
                      multiprocessing_method='joblib'
                      )


def test_zero_negative_pvalue():
    elements = ['a', 'b', 'c']

    def one(complements):
        return 1

    for_bootstrap, _, _ = msa.interface(elements=elements,
                                        n_permutations=100,
                                        objective_function=one)
    with pytest.raises(ValueError):
        ut.bootstrap_hypothesis_testing(dataset=for_bootstrap,
                                        p_value=-1)

        ut.bootstrap_hypothesis_testing(dataset=for_bootstrap,
                                        p_value=0)
