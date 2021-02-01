import numpy as np
from pandora.quantized_constants import C1_MAX, C2_MAX, C3_MAX, C4_MAX, C5_MAX, C6_MAX, \
    C7_MAX, C8_MAX, H1_MAX, H2_MAX, H3_MAX, H6_MAX

RNG = np.random.default_rng()


class PermutingGenerator:

    def __init__(self, n_days: int):
        self._n_days = n_days
        if self._n_days > 2:
            segment_main_length = int(n_days / 3)
            segment_tail_length = n_days - (2 * segment_main_length)
            self._segment_main = PermutingGenerator.permute_segment(segment_main_length)
            if segment_main_length == segment_tail_length:
                self._segment_tail = self._segment_main
            else:
                self._segment_tail = PermutingGenerator.permute_segment(segment_tail_length)
        else:
            self._segment_main = PermutingGenerator.permute_segment(n_days)
            self._segment_tail = None

    def generate_sequence(self) -> [int]:
        if self._n_days > 2:
            return [*RNG.choice(self._segment_main),
                    *RNG.choice(self._segment_main),
                    *RNG.choice(self._segment_tail)]
        else:
            return [*RNG.choice(self._segment_main)]

    @staticmethod
    def permute_segment(n_days):
        initial_min = n_days * 12 * [0]
        initial_one = n_days * 12 * [1]
        initial_two = n_days * 12 * [2]
        initial_max = n_days * [C1_MAX, C2_MAX, C3_MAX, C4_MAX,
                                C5_MAX, C6_MAX, C7_MAX, C8_MAX,
                                H1_MAX, H2_MAX, H3_MAX, H6_MAX]
        permutations = [initial_min,
                        initial_one,
                        initial_two,
                        initial_max]
        permutations += PermutingGenerator.permute_npi(C1_MAX, 0, n_days)
        permutations += PermutingGenerator.permute_npi(C2_MAX, 1, n_days)
        permutations += PermutingGenerator.permute_npi(C3_MAX, 2, n_days)
        permutations += PermutingGenerator.permute_npi(C4_MAX, 3, n_days)
        permutations += PermutingGenerator.permute_npi(C5_MAX, 4, n_days)
        permutations += PermutingGenerator.permute_npi(C6_MAX, 5, n_days)
        permutations += PermutingGenerator.permute_npi(C7_MAX, 6, n_days)
        permutations += PermutingGenerator.permute_npi(C8_MAX, 7, n_days)
        permutations += PermutingGenerator.permute_npi(H1_MAX, 8, n_days)
        permutations += PermutingGenerator.permute_npi(H2_MAX, 9, n_days)
        permutations += PermutingGenerator.permute_npi(H3_MAX, 10, n_days)
        permutations += PermutingGenerator.permute_npi(H6_MAX, 11, n_days)
        return permutations

    @staticmethod
    def permute_npi(npi_max: int,
                    npi_position: int,
                    n_days: int):
        permutations = []
        for i in range(0, npi_max + 1):
            if i > 0:
                x = n_days * [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                x[npi_position] = i
                permutations.append(x)
            if i < C1_MAX:
                x = n_days * [C1_MAX, C2_MAX, C3_MAX, C4_MAX,
                              C5_MAX, C6_MAX, C7_MAX, C8_MAX,
                              H1_MAX, H2_MAX, H3_MAX, H6_MAX]
                x[npi_position] = i
                permutations.append(x)
        return permutations
