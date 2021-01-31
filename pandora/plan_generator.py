import numpy as np

from pandora.quantized_constants import NPI_LIMITS, C1_MAX, C2_MAX, C3_MAX, C4_MAX, C5_MAX, C6_MAX, \
    C7_MAX, C8_MAX, H1_MAX, H2_MAX, H3_MAX, H6_MAX

RNG = np.random.default_rng()


class Plan:
    def __init__(self, days: int, actions: [int]) -> None:
        self.actions = actions
        self.days = days
        self.stringency = None
        self.estimated_cases = None
        self.score = None

    def update_stringency(self, factors: [float]) -> None:
        stringency = 0.
        for i, v in enumerate(self.actions):
            stringency += float(v) * factors[i]
        self.stringency = stringency


def generate_random_plan(days) -> Plan:
    # set the random values, until the tail is reached
    # note, we don't need to explicitly set the tail values to 0, since they will already be set
    days_tail = 14
    days_random = max(0, days - days_tail)
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
    values = [0] * 12 * days
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
    return Plan(days=days, actions=values)


def generate_plans(n_indexes: int,
                   n_candidates_per_index: int,
                   days: int,
                   factors: [float],
                   limits: [int]) -> [[Plan]]:
    bin_size = len(NPI_LIMITS) / (float(n_indexes) * float(n_candidates_per_index))
    bin_index = 0
    indexes = np.empty(n_indexes, dtype=object)
    for i in range(n_indexes):
        candidates = np.empty(n_candidates_per_index, dtype=object)
        for j in range(n_candidates_per_index):
            bin_min = bin_size * bin_index
            bin_max = bin_size * (bin_index + 1)
            bin_index += 1
            candidates[j] = generate_plan(days, factors, limits, bin_min, bin_max)
        indexes[i] = candidates
    return indexes


def generate_plan(days: int,
                  factors: [float],
                  limits: [int],
                  bin_min: float,
                  bin_max: float) -> Plan:
    plan = generate_random_plan(days)
    plan.update_stringency(factors)

    # the cost of the plan is TOO HIGH and must be DECREASED
    stringency = plan.stringency
    actions = plan.actions
    if stringency > bin_max:
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
        RNG.shuffle(modifiable_actions)

        # decrementing loop
        for action_index in modifiable_actions:
            actions[action_index] -= 1
            stringency -= factors[action_index]
            if stringency <= bin_max:
                plan.stringency = stringency
                break

    # the cost of the plan is TOO LOW and must be INCREASED
    elif stringency < bin_min:

        # for every plan element, add its index to a list if it is NON-MAX
        # first, determine how large the array will be
        counter = 0
        for action_index, v in enumerate(actions):
            counter += limits[action_index] - v

        # fill in the array with the positions
        modifiable_actions = [0] * counter
        modifiable_action_index = 0
        for action_index, v in enumerate(actions):
            v_add = limits[action_index] - v
            for _ in range(v_add):
                modifiable_actions[modifiable_action_index] = action_index
                modifiable_action_index += 1
        RNG.shuffle(modifiable_actions)

        # increment loop
        for action_index in modifiable_actions:
            actions[action_index] += 1
            stringency += factors[action_index]
            if stringency >= bin_min:
                plan.stringency = stringency
                break
    return plan


"""

w = [1.000000] * len(NPI_LIMITS) * 90
lim = NPI_LIMITS * 90
fac = w.copy()
for fac_i in range(len(NPI_LIMITS) * 90):
    fac[fac_i] = w[fac_i] / lim[fac_i] / float(90)

start = time.time_ns()
my_plans = generate_plans(PLANS,
                          CANDIDATES_PER_PLAN,
                          90,
                          fac,
                          lim)
stop = time.time_ns()
print(len(my_plans[0]))
time_total = float(stop - start) / 1e9
print(f"time was {time_total}")
"""
