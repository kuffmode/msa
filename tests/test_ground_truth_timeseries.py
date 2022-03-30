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


def test_contributions():
    data, final_wave, elements = prepare_data()

    def score_function(complements):
        return final_wave - data[complements, :].sum(0)

    shapley_table, _, _ = msa.interface(
        elements=elements,
        n_permutations=5_000,
        objective_function=score_function,
        n_parallel_games=-1)

    assert np.allclose(shapley_table.groupby(level=1).mean(), data.T)


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
