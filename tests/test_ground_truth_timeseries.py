from msapy import msa
from itertools import product

import numpy as np
import pytest


def generate_wave_data(amp_freq_pairs, timestamps, sampling_rate):
    frequencies = amp_freq_pairs[:, 1, None]
    amplitudes = amp_freq_pairs[:, 0, None]
    timestamps = np.broadcast_to(
        timestamps, (amplitudes.shape[0], sampling_rate))
    data = np.sin(2 * np.pi * timestamps * frequencies) * amplitudes
    return data


@pytest.mark.parametrize("n_parallel_games", [1, -1])
def test_contributions(n_parallel_games):
    data, final_wave, elements = prepare_data()

    def score_function(complements):
        return final_wave - data[complements, :].sum(0)

    shapley_table = msa.interface(
        elements=elements,
        n_permutations=5_000,
        objective_function=score_function,
        n_parallel_games=n_parallel_games,
        save_permutations=True)

    assert np.allclose(shapley_table.groupby(level=1).mean(), data.T)


@pytest.mark.parametrize("n_cores, multiprocessing_method, parallelize_over_games",
                         [(1, 'joblib', True), (-1, 'joblib', True), (1, 'joblib', False), (-1, 'joblib', False)])
def test_estimate_causal_influence(n_cores, multiprocessing_method, parallelize_over_games):
    true_causal_influences = np.random.normal(0, 5, (4, 4, 100))

    true_causal_influences[np.diag_indices(4)] = 0

    def objective_function_causal_influence(complements, index):
        return true_causal_influences[index].sum(0) - true_causal_influences[index, complements].sum(0)

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


def prepare_data():
    sampling_rate = 200
    sampling_interval = 1/sampling_rate
    timestamps = np.arange(0, 1, sampling_interval)

    frequencies = np.arange(1, 10, 1.5)
    amplitudes = np.arange(0.2, 2, 0.4)

    amp_freq_pairs = np.array(
        list(map(list, product(amplitudes, frequencies))))

    data = generate_wave_data(amp_freq_pairs, timestamps, sampling_rate)
    final_wave = data.sum(0)
    elements = list(range(len(data)))
    return data, final_wave, elements
