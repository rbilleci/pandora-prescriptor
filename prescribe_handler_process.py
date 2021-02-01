import logging
import os
import time
from datetime import timedelta
from logging import info

import numpy as np
import pandas as pd

import pandora.quantized_predictor
from covid_xprize.standard_predictor.xprize_predictor import NB_LOOKBACK_DAYS, WINDOW_SIZE
from pandora.prescription_generator import PrescriptionGenerator
from pandora.quantized_constants import NPI_LIMITS, C1, C2, C3, C4, C5, C6, C7, C8, H1, H2, H3, H6

if not os.path.exists('logs'):
    os.makedirs('logs')
logging.basicConfig(level=logging.INFO, format='%(asctime)s\t%(levelname)s\t%(filename)s\t%(message)s')
logging.getLogger().handlers = [logging.FileHandler(f"logs/prescribe-{time.strftime('%Y-%m-%d')}"),
                                logging.StreamHandler()]

quantized_predictor = pandora.quantized_predictor.QuantizedPredictor()


def prescribe_loop_for_geo(geo: str,
                           start_date,
                           costs,
                           past_cases: np.ndarray,
                           past_ips: np.ndarray,
                           n_days: int,
                           limits: [int],
                           population: int,
                           prescription_candidates_per_index: int,
                           prescription_generator: PrescriptionGenerator) -> pd.DataFrame:
    info(f"{geo} searching for prescriptions...")
    loop_time_start = time.time_ns()
    factors = costs.tolist() * n_days
    for i, f in enumerate(factors):
        factors[i] = f / limits[i] / n_days

    # get the input
    df = compute_context(geo, past_cases, population)
    past_actions = resolve_past_actions(past_ips)
    past_context = resolve_past_context(df)

    # get the candidate prescriptions
    prescriptions_by_index = prescription_generator.generate_prescriptions(prescription_candidates_per_index, factors)

    # evaluate, score, and get the best prescriptions for each prescription index
    evaluate(df, n_days, past_actions, past_context, prescriptions_by_index)

    # score, and determine the best prescription
    score(prescriptions_by_index)
    best_prescriptions = resolve_best_prescriptions(geo, prescriptions_by_index)

    # get the result, log the timing, and return
    df_result = generate_dataframe(geo, n_days, start_date, best_prescriptions)
    loop_time_end = time.time_ns()
    info(f"{geo} took {(loop_time_end - loop_time_start) / 1e9} seconds")
    return df_result


def resolve_past_actions(past_ips):
    return past_ips[-NB_LOOKBACK_DAYS:]


def resolve_past_context(df):
    past_context = df['PredictionRatio'].values[-NB_LOOKBACK_DAYS:]
    try:
        return np.reshape(past_context, (NB_LOOKBACK_DAYS, 1))
    except ValueError:
        return [[0] * len(NPI_LIMITS)] * NB_LOOKBACK_DAYS


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


def evaluate(df,
             n_days: int,
             past_actions,
             past_context,
             prescriptions_by_index):
    for prescriptions_for_index in prescriptions_by_index:
        for i, prescription in enumerate(prescriptions_for_index):
            prescription.estimated_cases = quantized_predictor.predict_geo(
                df,
                n_days,
                past_context,
                past_actions,
                np.reshape(prescription.actions, (n_days, 12)))


def score(prescriptions_by_index):
    for prescription_index, prescriptions in enumerate(prescriptions_by_index):
        # NOTE: we could optimize this loop by sorting by estimated cases, then stringency
        # so we could quickly break as soon as we are finding we are not dominating on cases
        for i, dominator in enumerate(prescriptions):
            for j, target in enumerate(prescriptions):
                if i == j:
                    continue
                if dominator.estimated_cases <= target.estimated_cases and dominator.stringency < target.stringency:
                    dominator.score = dominator.score + 1.


def resolve_best_prescriptions(geo, prescriptions_by_index):
    best_prescriptions = []
    for i, prescriptions in enumerate(prescriptions_by_index):
        best = None
        for prescription in prescriptions:
            if best is None:
                best = prescription
            elif prescription.score > best.score:
                best = prescription
            elif prescription.score == best.score and prescription.estimated_cases < best.estimated_cases:
                best = prescription
        info(f"{geo} index [{i}] - {best.estimated_cases} {best.stringency} {best.score}")
        best_prescriptions.append(best)
    return best_prescriptions


def generate_dataframe(geo,
                       n_days,
                       start_date,
                       best_prescriptions):
    country_name = geo.split('__')[0]
    region_name = geo.split('__')[1]
    data = []
    for prescription_index, prescription in enumerate(best_prescriptions):
        date = pd.to_datetime(start_date)
        actions = prescription.actions
        for d in range(n_days):
            data.append([prescription_index,
                         country_name,
                         region_name,
                         date,
                         actions[12 * d + 0],
                         actions[12 * d + 1],
                         actions[12 * d + 2],
                         actions[12 * d + 3],
                         actions[12 * d + 4],
                         actions[12 * d + 5],
                         actions[12 * d + 6],
                         actions[12 * d + 7],
                         actions[12 * d + 8],
                         actions[12 * d + 9],
                         actions[12 * d + 10],
                         actions[12 * d + 11],
                         prescription.estimated_cases,
                         prescription.score])
            date += timedelta(days=1)
    return pd.DataFrame(data=data, columns=['PrescriptionIndex',
                                            'CountryName',
                                            'RegionName',
                                            'Date',
                                            C1, C2, C3, C4, C5, C6, C7, C8, H1, H2, H3, H6,
                                            'EstimatedCases',
                                            'EstimatedScore'])
