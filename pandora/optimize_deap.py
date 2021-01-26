import os
import random
import sys
import pyswarms as ps
from datetime import datetime
import skopt
from skopt.space import Integer

import quantized_predictor
import pandas as pd
import numpy as np
from scipy.optimize import minimize, dual_annealing, basinhopping, shgo, differential_evolution
from scipy.optimize import Bounds

from pandora.quantized_constants import START_DATE, END_DATE, \
    C1, C2, C3, C1_MAX, C2_MAX, C3_MAX, C4_MAX, C5_MAX, C6_MAX, C7_MAX, C8_MAX, \
    H1_MAX, H2_MAX, H3_MAX, H6_MAX, C4, C5, C6, C7, C8, H1, H2, H3, H6

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_INPUT_FILE = os.path.join(ROOT_DIR, "../covid_xprize", "validation/data/future_ip.csv")


def optimize(start_date_str: str,
             end_date_str: str):
    predictor = quantized_predictor.QuantizedPredictor()
    start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d')
    end_date = pd.to_datetime(end_date_str, format='%Y-%m-%d')
    number_of_days = (end_date - start_date).days + 1
    npis_df = predictor.load_original_data(EXAMPLE_INPUT_FILE)
    geos = npis_df.GeoID.unique()
    truncated_df = predictor.df[predictor.df.Date < START_DATE]
    country_samples = predictor._create_country_samples(truncated_df, geos, False)

    for geo in geos:
        df_geo = truncated_df[truncated_df.GeoID == geo]
        geo_start_date = min(df_geo.Date.max() + np.timedelta64(1, 'D'), start_date)
        df_geo_npis = npis_df[(npis_df.Date >= geo_start_date) & (npis_df.Date <= end_date)]
        df_geo_npis = df_geo_npis[df_geo_npis.GeoID == geo]
        optimize_with_pyswarms(predictor,
                               geo,
                               df_geo,
                               df_geo_npis,
                               number_of_days,
                               country_samples)


def generate_initial_points(number_of_days):
    # TODO: these plans can just be loaded... we don't need to generate them.
    if number_of_days == 1:
        return generate_initial_points_for_segment(number_of_days)
    else:
        segment_1 = generate_initial_points_for_segment(int(number_of_days / 2))
        segment_2 = generate_initial_points_for_segment(number_of_days - int(number_of_days / 2))
        joined = []
        for x1 in segment_1:
            for x2 in segment_2:
                joined.append(x1 + x2)
        return joined


def generate_initial_points_for_segment(number_of_days):
    initial_min = number_of_days * 12 * [0]
    initial_one = number_of_days * 12 * [1]
    initial_two = number_of_days * 12 * [2]
    initial_max = number_of_days * [C1_MAX, C2_MAX, C3_MAX, C4_MAX,
                                    C5_MAX, C6_MAX, C7_MAX, C8_MAX,
                                    H1_MAX, H2_MAX, H3_MAX, H6_MAX]
    initial_points = [initial_min,
                      initial_one,
                      initial_two,
                      initial_max]
    initial_points += generate_initial_points_for_npi(C1_MAX, 0, number_of_days)
    initial_points += generate_initial_points_for_npi(C2_MAX, 1, number_of_days)
    initial_points += generate_initial_points_for_npi(C3_MAX, 2, number_of_days)
    initial_points += generate_initial_points_for_npi(C4_MAX, 3, number_of_days)
    initial_points += generate_initial_points_for_npi(C5_MAX, 4, number_of_days)
    initial_points += generate_initial_points_for_npi(C6_MAX, 5, number_of_days)
    initial_points += generate_initial_points_for_npi(C7_MAX, 6, number_of_days)
    initial_points += generate_initial_points_for_npi(C8_MAX, 7, number_of_days)
    initial_points += generate_initial_points_for_npi(H1_MAX, 8, number_of_days)
    initial_points += generate_initial_points_for_npi(H2_MAX, 9, number_of_days)
    initial_points += generate_initial_points_for_npi(H3_MAX, 10, number_of_days)
    initial_points += generate_initial_points_for_npi(H6_MAX, 11, number_of_days)
    return initial_points


def generate_initial_points_for_npi(npi_max: int,
                                    npi_position: int,
                                    number_of_days: int):
    initial_points = []
    for i in range(0, npi_max + 1):
        if i > 0:
            x = number_of_days * [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            x[npi_position] = i
            initial_points.append(x)
        if i < C1_MAX:
            x = number_of_days * [C1_MAX, C2_MAX, C3_MAX, C4_MAX,
                                  C5_MAX, C6_MAX, C7_MAX, C8_MAX,
                                  H1_MAX, H2_MAX, H3_MAX, H6_MAX]
            x[npi_position] = i
            initial_points.append(x)
    return initial_points


def optimize_with_skopt(predictor: quantized_predictor.QuantizedPredictor,
                        geo,
                        df_geo: pd.DataFrame,
                        df_geo_npis: pd.DataFrame,
                        number_of_days,
                        country_samples):
    def objective_function(values):
        df_opt = df_geo_npis.copy()
        row_number = 0
        for index, _ in df_opt.iterrows():
            df_opt.loc[index, [C1, C2, C3, C4, C5, C6, C7, C8, H1, H2, H3, H6]] = \
                values[(row_number * 12):(row_number * 12) + 12]
            row_number += 1

        cases = predictor.predict_geo(geo, df_geo, df_opt, number_of_days, country_samples)
        print(f"{geo} {cases}")
        return cases

    # get the initial points, then take a random sample,
    # since we will not have the time budget to perform too many computations
    initial_points = generate_initial_points(number_of_days)
    initial_points = random.sample(initial_points, 10)

    # and, generate a list of completely random points
    random_points = 2

    search_space = list()
    for i in range(0, number_of_days):
        search_space.append(Integer(0, C1_MAX, name='C1'))
        search_space.append(Integer(0, C2_MAX, name='C2'))
        search_space.append(Integer(0, C3_MAX, name='C3'))
        search_space.append(Integer(0, C4_MAX, name='C4'))
        search_space.append(Integer(0, C5_MAX, name='C5'))
        search_space.append(Integer(0, C6_MAX, name='C6'))
        search_space.append(Integer(0, C7_MAX, name='C7'))
        search_space.append(Integer(0, C8_MAX, name='C8'))
        search_space.append(Integer(0, H1_MAX, name='H1'))
        search_space.append(Integer(0, H2_MAX, name='H2'))
        search_space.append(Integer(0, H3_MAX, name='H3'))
        search_space.append(Integer(0, H6_MAX, name='H6'))

    res = skopt.gp_minimize(objective_function,
                            x0=initial_points,
                            xi=0.01 / 10.,
                            kappa=1.96 / 10.,
                            n_calls=len(initial_points) + random_points + 3,
                            n_initial_points=random_points,
                            dimensions=search_space)

    print(res)
    return None


# gets to about 900
def optimize_with_pyswarms(predictor: quantized_predictor.QuantizedPredictor,
                           geo,
                           df_geo: pd.DataFrame,
                           df_geo_npis: pd.DataFrame,
                           number_of_days,
                           country_samples):
    def objective_function(particle_list):

        costs = np.zeros([len(particle_list)])
        for x in range(0, len(particle_list)):
            values = particle_list[x]

            # round the values
            # print(values[0], values[1], values[2])
            for i in range(0, len(values)):
                values[i] = round(values[i])

            df_opt = df_geo_npis.copy()
            row_number = 0
            for index, _ in df_opt.iterrows():
                df_opt.loc[index, [C1, C2, C3, C4, C5, C6, C7, C8, H1, H2, H3, H6]] = \
                    values[(row_number * 12):(row_number * 12) + 12]
                row_number += 1
            cases = predictor.predict_geo(geo, df_geo, df_opt, number_of_days, country_samples)
            costs[x] = cases
        return costs

    print(geo)
    bounds_lower = number_of_days * [0] * 12
    bounds_upper = number_of_days * [C1_MAX, C2_MAX, C3_MAX, C4_MAX,
                                     C5_MAX, C6_MAX, C7_MAX, C8_MAX,
                                     H1_MAX, H2_MAX, H3_MAX, H6_MAX]
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.8}

    # get the initial points, then take a random sample,
    # since we will not have the time budget to perform too many computations
    n_particles = 50
    initial_points = generate_initial_points(number_of_days)
    initial_points = random.sample(initial_points, n_particles)
    initial_points = np.array(initial_points).astype('float')

    optimizer = ps.single.GlobalBestPSO(n_particles=n_particles,
                                        init_pos=initial_points,
                                        dimensions=12 * number_of_days,
                                        bounds=(bounds_lower, bounds_upper),
                                        options=options)

    cost, pos = optimizer.optimize(objective_function, iters=2)
    print(cost)


def optimize_with_scipy(predictor: quantized_predictor.QuantizedPredictor,
                        geo,
                        df_geo: pd.DataFrame,
                        df_geo_npis: pd.DataFrame,
                        number_of_days,
                        country_samples):
    def objective_function(values):
        # assign the values
        df_opt = df_geo_npis.copy()
        row_number = 0
        for index, _ in df_opt.iterrows():
            df_opt.loc[index, [C1, C2, C3, C4, C5, C6, C7, C8, H1, H2, H3, H6]] = \
                values[(row_number * 12):(row_number * 12) + 12]
            row_number += 1

        cases = predictor.predict_geo(geo, df_geo, df_opt, number_of_days, country_samples)
        print(f"{geo} {cases}")
        return cases

    bounds_lower = number_of_days * [0.] * 12
    bounds_upper = number_of_days * [C1_MAX, C2_MAX, C3_MAX, C4_MAX, C5_MAX, C6_MAX, C7_MAX, C8_MAX,
                                     H1_MAX, H2_MAX, H3_MAX, H6_MAX]
    bounds = Bounds(bounds_lower, bounds_upper)

    # get the initial points, then take a random sample,
    # since we will not have the time budget to perform too many computations
    initial_points = generate_initial_points(number_of_days)
    initial_points = random.sample(initial_points, 500)

    # and, generate a list of completely random points
    random_points = 100

    # optimization_result = minimize(objective_function, method='SLSQP', bounds=bounds)

    # res = dual_annealing(objective_function, initial_points=initial_points,
    #                     bounds=list(zip(bounds_lower, bounds_upper)), seed=1233)
    # res = shgo(objective_function, initial_points, bounds=list(zip(bounds_lower, bounds_upper)))

    """
    res = shgo(objective_function,
               n=10,
               iters=1,
               # sampling_method='sobol',
               bounds=list(zip(bounds_lower, bounds_upper)),
               minimizer_kwargs={"method": "L-BFGS-B",
                                 "bounds": bounds,
                                 "maxiter": 100})
    """

    # print(res)
    return None


print(f"{datetime.now()} - optimizing")
optimize(START_DATE, END_DATE)
