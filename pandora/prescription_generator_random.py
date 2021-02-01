import numpy as np

from pandora.quantized_constants import C1_MAX, C2_MAX, C3_MAX, C4_MAX, C5_MAX, C6_MAX, \
    C7_MAX, C8_MAX, H1_MAX, H2_MAX, H3_MAX, H6_MAX

RNG = np.random.default_rng()


class RandomGenerator:

    def __init__(self, n_days: int):
        self._n_days = n_days

    def generate_sequence(self) -> [int]:
        # set the random values, until the tail is reached
        # note, we don't need to explicitly set the tail values to 0, since they will already be set
        days_tail = 14
        days_random = max(0, self._n_days - days_tail)
        values_c1 = RNG.integers(0, C1_MAX + 1, size=days_random)
        values_c2 = RNG.integers(0, C2_MAX + 1, size=days_random)
        values_c3 = RNG.integers(0, C3_MAX + 1, size=days_random)
        values_c4 = RNG.integers(0, C4_MAX + 1, size=days_random)
        values_c5 = RNG.integers(0, C5_MAX + 1, size=days_random)
        values_c6 = RNG.integers(0, C6_MAX + 1, size=days_random)
        values_c7 = RNG.integers(0, C7_MAX + 1, size=days_random)
        values_c8 = RNG.integers(0, C8_MAX + 1, size=days_random)
        values_h1 = RNG.integers(0, H1_MAX + 1, size=days_random)
        values_h2 = RNG.integers(0, H2_MAX + 1, size=days_random)
        values_h3 = RNG.integers(0, H3_MAX + 1, size=days_random)
        values_h6 = RNG.integers(0, H6_MAX + 1, size=days_random)
        values = [0] * 12 * self._n_days
        for d in range(days_random):
            day_index = d * 12
            values[day_index + 0] = values_c1[d]
            values[day_index + 1] = values_c2[d]
            values[day_index + 2] = values_c3[d]
            values[day_index + 3] = values_c4[d]
            values[day_index + 4] = values_c5[d]
            values[day_index + 5] = values_c6[d]
            values[day_index + 6] = values_c7[d]
            values[day_index + 7] = values_c8[d]
            values[day_index + 8] = values_h1[d]
            values[day_index + 9] = values_h2[d]
            values[day_index + 10] = values_h3[d]
            values[day_index + 11] = values_h6[d]
        return values
