import random

import numpy as np

from pandora.prescription_generator_permuted import PermutingGenerator
from pandora.prescription_generator_random import RandomGenerator
from pandora.quantized_constants import NPI_LIMITS, PRESCRIPTION_INDEXES


class Prescription:
    def __init__(self, actions: [int]) -> None:
        self.actions = actions
        self.stringency = None
        self.estimated_cases = None
        self.score = 0.

    def update_stringency(self, factors: [float]) -> None:
        stringency = 0.
        for i, v in enumerate(self.actions):
            stringency += float(v) * factors[i]
        self.stringency = stringency


class PrescriptionGenerator:

    def __init__(self,
                 n_days: int,
                 limits: [int]) -> None:
        self._rng = np.random.default_rng()
        self._n_days = n_days
        self._limits = limits
        self._permuting_generator = PermutingGenerator(n_days)
        self._random_generator = RandomGenerator(n_days)

    def generate_prescriptions(self,
                               prescription_candidates_per_index,
                               factors):
        bin_size = len(NPI_LIMITS) / (float(PRESCRIPTION_INDEXES) * float(prescription_candidates_per_index))
        bin_index = 0
        indexes = np.empty(PRESCRIPTION_INDEXES, dtype=object)
        for i in range(PRESCRIPTION_INDEXES):
            candidates = np.empty(prescription_candidates_per_index, dtype=object)
            for j in range(prescription_candidates_per_index):
                bin_min = bin_size * bin_index
                bin_max = bin_size * (bin_index + 1)
                bin_index += 1
                candidates[j] = self.generate_prescription(
                    factors,
                    bin_min,
                    bin_max)
            indexes[i] = candidates
        return indexes

    def generate_prescription(self,
                              factors,
                              bin_min,
                              bin_max):
        if random.random() < 0.50:
            prescription = Prescription(self._permuting_generator.generate_sequence())
            self.fit(prescription, factors, bin_min, bin_max)
            return prescription
        else:
            prescription = Prescription(self._random_generator.generate_sequence())
            self.fit(prescription, factors, bin_min, bin_max)
            return prescription

    def fit(self,
            prescription: Prescription,
            factors: [float],
            bin_min: float,
            bin_max: float) -> None:
        prescription.update_stringency(factors)
        stringency = prescription.stringency
        if stringency > bin_max:
            self.fit_minimize(prescription, factors, bin_min)
        elif stringency < bin_min:
            self.fit_maximize(prescription, factors, bin_min)

    def fit_minimize(self,
                     prescription: Prescription,
                     factors: [float],
                     bin_max: float):
        actions = prescription.actions
        stringency = prescription.stringency
        # for every plan element, add its index to a list if it is NON-ZERO
        # first, determine how large the array will be
        counter = 0
        for v in actions:
            counter += v

        # fill in the array with the positions
        modifiable_actions = [0] * counter
        modifiable_action_index = 0
        for action_index, v in enumerate(actions):
            for _ in range(v):
                modifiable_actions[modifiable_action_index] = action_index
                modifiable_action_index += 1
        self._rng.shuffle(modifiable_actions)

        # decrementing loop
        for action_index in modifiable_actions:
            actions[action_index] -= 1
            stringency -= factors[action_index]
            if stringency <= bin_max:
                prescription.stringency = stringency
                return

    def fit_maximize(self,
                     prescription: Prescription,
                     factors: [float],
                     bin_min: float):
        actions = prescription.actions
        stringency = prescription.stringency
        # for every plan element, add its index to a list if it is NON-MAX
        # first, determine how large the array will be
        counter = 0
        for action_index, v in enumerate(actions):
            counter += self._limits[action_index] - v

        # fill in the array with the positions
        modifiable_actions = [0] * counter
        modifiable_action_index = 0
        for action_index, v in enumerate(actions):
            v_add = self._limits[action_index] - v
            for _ in range(v_add):
                modifiable_actions[modifiable_action_index] = action_index
                modifiable_action_index += 1
        self._rng.shuffle(modifiable_actions)

        # increment loop
        for action_index in modifiable_actions:
            actions[action_index] += 1
            stringency += factors[action_index]
            if stringency >= bin_min:
                prescription.stringency = stringency
                return
