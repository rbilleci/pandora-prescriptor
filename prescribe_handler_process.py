import logging
import os
import time

import numpy as np
import pandas as pd

import pandora.quantized_predictor
from covid_xprize.standard_predictor.xprize_predictor import NB_LOOKBACK_DAYS, WINDOW_SIZE
from pandora import plan_generator

if not os.path.exists('logs'):
    os.makedirs('logs')
logging.basicConfig(level=logging.INFO, format='%(asctime)s\t%(levelname)s\t%(filename)s\t%(message)s')
logging.getLogger().handlers = [logging.FileHandler(f"logs/prescribe-{time.strftime('%Y-%m-%d')}"),
                                logging.StreamHandler()]

quantized_predictor = pandora.quantized_predictor.QuantizedPredictor()


def prescribe_loop_for_geo(geo: str,
                           costs,
                           past_cases: np.ndarray,
                           past_ips: np.ndarray,
                           n_days: int,
                           limits: [int],
                           population: int):
    factors = costs.tolist() * n_days
    for i, f in enumerate(factors):
        factors[i] = f / limits[i] / n_days

    # get the input
    df = compute_context(geo, past_cases, population)
    past_actions = past_ips[-NB_LOOKBACK_DAYS:]
    past_context = df['PredictionRatio'].values[-NB_LOOKBACK_DAYS:]
    past_context = np.reshape(past_context, (NB_LOOKBACK_DAYS, 1))

    plan_ranges = plan_generator.generate_plans(10, 10, n_days, factors, limits)
    for candidates in plan_ranges:
        find_best_plan(geo,
                       candidates,
                       n_days,
                       df,
                       past_context,
                       past_actions)

    return geo, costs


def find_best_plan(geo,
                   candidates,
                   n_days,
                   df,
                   past_context,
                   past_actions):
    best_plan = None
    best_plan_estimated_cases = 1e12
    best_plan_estimated_cost = 1e12
    for candidate in candidates:
        plan = np.reshape(candidate[0], (n_days, 12))
        estimated_cost = candidate[1]
        estimated_cases = quantized_predictor.predict_geo(df,
                                                          n_days,
                                                          past_context,
                                                          past_actions,
                                                          plan)
        if estimated_cases < best_plan_estimated_cases:
            best_plan = plan
            best_plan_estimated_cases = estimated_cases
            best_plan_estimated_cost = estimated_cost
    print(f"{geo} - {best_plan_estimated_cases} {best_plan_estimated_cost}")
    return best_plan, best_plan_estimated_cases, best_plan_estimated_cost


def compute_context(geo: str,
                    past_cases: np.ndarray,
                    population: int):
    past_cases = past_cases.reshape(1, -1)
    df = pd.DataFrame({'NewCases': past_cases[0]})
    df['NewCases'] = df['NewCases'].clip(lower=0).fillna(0)
    df['ConfirmedCases'] = df['NewCases'].cumsum().fillna(0)
    df['GeoID'] = geo
    df['Population'] = population
    df['SmoothNewCases'] = df['NewCases'].rolling(WINDOW_SIZE, center=False).mean().fillna(0).reset_index(0, drop=True)
    df['CaseRatio'] = df['SmoothNewCases'].pct_change().fillna(0).replace(np.inf, 0) + 1
    df['ProportionInfected'] = df['ConfirmedCases'] / df['Population']
    df['PredictionRatio'] = df['CaseRatio'] / (1 - df['ProportionInfected'])
    return df
