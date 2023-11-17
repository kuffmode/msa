from msapy import msa
import pytest
import numpy as np


def create_mask(image_size, grid_size, box):
    x = box // grid_size
    y = box % grid_size

    start_x = x * image_size // grid_size
    end_x = start_x + image_size // grid_size

    start_y = y * image_size // grid_size
    end_y = start_y + image_size // grid_size

    mask = np.ones((image_size, image_size), bool)
    mask[start_x:end_x, start_y:end_y] = False

    return mask


def mask_image(image, mask):
    masked_image = image.copy()
    masked_image[mask] = 0
    return masked_image


np.random.seed(0)
image = (np.random.random(size=(10, 10, 3)))
contributions = np.array(
    [mask_image(image, create_mask(10, 2, i)) for i in range(16)])


def objective_func(complements):
    return (image - contributions[complements, :].sum(0))


@pytest.mark.parametrize("n_parallel_games, lazy", [[1, True], [-1, True], [1, False], [-1, False]])
def test_contributions(n_parallel_games, lazy):
    shapley_mode = msa.interface(
        elements=list(range(4)),
        n_permutations=100,
        objective_function=objective_func,
        n_parallel_games=n_parallel_games,
        lazy=lazy
    )

    assert np.allclose(shapley_mode.get_total_contributions(), image)


@pytest.mark.parametrize("n_parallel_games, lazy", [[1, True], [-1, True], [1, False], [-1, False]])
def test_contributions_permutations(n_parallel_games, lazy):
    shapley_table_nd = msa.interface(
        elements=list(range(4)),
        n_permutations=100,
        objective_function=objective_func,
        n_parallel_games=n_parallel_games,
        save_permutations=True,
        lazy=lazy
    )

    assert np.allclose(shapley_table_nd.shapley_modes.get_total_contributions(), image)


@pytest.mark.parametrize("n_cores, multiprocessing_method, parallelize_over_games",
                         [(1, 'joblib', True), (-1, 'joblib', True), (1, 'joblib', False), (-1, 'joblib', False)])
def test_estimate_causal_influence(n_cores, multiprocessing_method, parallelize_over_games):
    true_causal_influences = np.random.normal(0, 5, (4, 4, 100))

    true_causal_influences[np.diag_indices(4)] = 0

    def objective_function_causal_influence(complements, index):
        return true_causal_influences[index].sum(0).reshape(2, 2, 5, 5) - true_causal_influences[index, complements].sum(0).reshape(2, 2, 5, 5)

    estimated_causal_influences = msa.estimate_causal_influences(elements=list(range(4)),
                                                                 n_permutations=10000,
                                                                 objective_function=objective_function_causal_influence,
                                                                 multiprocessing_method=multiprocessing_method,
                                                                 parallelize_over_games=parallelize_over_games,
                                                                 n_cores=n_cores)

    estimated_causal_influences = estimated_causal_influences.groupby(
        level=0).mean().values
    np.fill_diagonal(estimated_causal_influences, 0)

    assert np.allclose(estimated_causal_influences,
                       true_causal_influences.mean(2))
