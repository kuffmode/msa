from core import msa
import pytest
import ray

elements = ['a', 'b', 'c']


def one(complements):
    return 1


@ray.remote
def two(complements):
    return 2


def test_negative_integers():
    with pytest.raises(ValueError):
        msa.make_permutation_space(elements=elements,
                                   n_permutations=-1)

        msa.interface(elements=elements,
                      n_permutations=-1,
                      objective_function=one,
                      )


def test_zero_permutations():
    with pytest.raises(ValueError):
        msa.make_permutation_space(elements=elements,
                                   n_permutations=0)

        msa.interface(elements=elements,
                      n_permutations=0,
                      objective_function=one,
                      )


def test_confusing_parallelstuff():
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
