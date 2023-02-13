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


@pytest.mark.parametrize("n_parallel_games, lazy", [[1, True], [-1, True], [1, False], [-1, False]])
def test_contributions(n_parallel_games, lazy):
    image = (np.random.random(size=(512, 512, 3)) * 255).astype(np.int16)
    contributions = [mask_image(image, create_mask(512, 4, i)) for i in range(16)]

    def objective_func(complements):
        contrib = image.copy()
        for i in complements:
            contrib -= contributions[i]

        return contrib.astype(np.int16)

    shapley_mode, _, _ = msa.interface(
        elements=list(range(16)),
        n_permutations=100,
        objective_function=objective_func,
        n_parallel_games=n_parallel_games,
        lazy=lazy
        )

    assert np.allclose(shapley_mode.get_total_contributions(), image)