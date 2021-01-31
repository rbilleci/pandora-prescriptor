from datetime import time

import numpy as np

from pandora.quantized_constants import NPI_LIMITS, C1_MAX, C2_MAX, C3_MAX, C4_MAX, C5_MAX, C6_MAX, \
    C7_MAX, C8_MAX, H1_MAX, H2_MAX, H3_MAX, H6_MAX

PLANS = 10
CANDIDATES_PER_PLAN = 100
RNG = np.random.default_rng()


def resolve_cost(plan: [int], factors: [float]) -> float:
    cost = 0.
    for i, v in enumerate(plan):
        cost += float(v) * factors[i]
    return cost


def generate_random_plan(days):
    values_c1 = RNG.integers(0, C1_MAX + 1, size=days)
    values_c2 = RNG.integers(0, C2_MAX + 1, size=days)
    values_c3 = RNG.integers(0, C3_MAX + 1, size=days)
    values_c4 = RNG.integers(0, C4_MAX + 1, size=days)
    values_c5 = RNG.integers(0, C5_MAX + 1, size=days)
    values_c6 = RNG.integers(0, C6_MAX + 1, size=days)
    values_c7 = RNG.integers(0, C7_MAX + 1, size=days)
    values_c8 = RNG.integers(0, C8_MAX + 1, size=days)
    values_h1 = RNG.integers(0, H1_MAX + 1, size=days)
    values_h2 = RNG.integers(0, H2_MAX + 1, size=days)
    values_h3 = RNG.integers(0, H3_MAX + 1, size=days)
    values_h6 = RNG.integers(0, H6_MAX + 1, size=days)
    values = [0] * 12 * days
    for d in range(days):
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


def generate_plans(n_plans: int,
                   n_candidates_per_plan: int,
                   days: int,
                   factors: [float],
                   limits: [int]) -> [([int], float)]:
    bin_size = len(NPI_LIMITS) / (float(n_plans) * float(n_candidates_per_plan))
    bin_index = 0
    plans = []
    for i in range(n_plans):
        candidates = []
        for j in range(n_candidates_per_plan):
            bin_min = bin_size * bin_index
            bin_max = bin_size * (bin_index + 1)
            bin_index += 1
            plan, cost = generate_plan(days, factors, limits, bin_min, bin_max)
            candidates.append([plan, cost])
        plans.append(candidates)
    return plans


def generate_plan(days: int,
                  factors: [float],
                  limits: [int],
                  bin_min: float,
                  bin_max: float) -> ([int], float):
    plan = generate_random_plan(days)
    cost = resolve_cost(plan, factors)

    # the cost of the plan is TOO HIGH and must be DECREASED
    if cost > bin_max:
        # for every plan element, add its index to a list if it is NON-ZERO
        # first, determine how large the array will be
        counter = 0
        for v in plan:
            counter += v

        # fill in the array with the positions
        npi = [0] * counter
        npi_index = 0
        for plan_index, v in enumerate(plan):
            for _ in range(v):
                npi[npi_index] = plan_index
                npi_index += 1
        RNG.shuffle(npi)

        # decrementing loop
        for npi_index, plan_index in enumerate(npi):
            plan[plan_index] -= 1
            cost -= factors[plan_index]
            if cost <= bin_max:
                break

    # the cost of the plan is TOO LOW and must be INCREASED
    elif cost < bin_min:

        # for every plan element, add its index to a list if it is NON-MAX
        # first, determine how large the array will be
        counter = 0
        for plan_index, v in enumerate(plan):
            counter += limits[plan_index] - v

        # fill in the array with the positions
        npi = [0] * counter
        npi_index = 0
        for plan_index, v in enumerate(plan):
            v_add = limits[plan_index] - v
            for _ in range(v_add):
                npi[npi_index] = plan_index
                npi_index += 1
        RNG.shuffle(npi)

        # increment loop
        for npi_index, plan_index in enumerate(npi):
            plan[plan_index] += 1
            cost += factors[plan_index]
            if cost >= bin_min:
                break
    return plan, cost


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
