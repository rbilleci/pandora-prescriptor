import logging
import os
import time
from datetime import timedelta
from logging import info, warning

import numpy as np
import pandas as pd

import pandora.quantized_predictor
from covid_xprize.standard_predictor.xprize_predictor import NB_LOOKBACK_DAYS, WINDOW_SIZE
from pandora import plan_generator
from pandora.quantized_constants import NPI_LIMITS, C1, C2, C3, C4, C5, C6, C7, C8, H1, H2, H3, H6

if not os.path.exists('logs'):
    os.makedirs('logs')
logging.basicConfig(level=logging.INFO, format='%(asctime)s\t%(levelname)s\t%(filename)s\t%(message)s')
logging.getLogger().handlers = [logging.FileHandler(f"logs/prescribe-{time.strftime('%Y-%m-%d')}"),
                                logging.StreamHandler()]
PRESCRIPTION_INDEXES = 10
PRESCRIPTION_CANDIDATES_PER_INDEX = 50

quantized_predictor = pandora.quantized_predictor.QuantizedPredictor()


def prescribe_loop_for_geo(geo: str,
                           start_date,
                           costs,
                           past_cases: np.ndarray,
                           past_ips: np.ndarray,
                           n_days: int,
                           limits: [int],
                           population: int) -> pd.DataFrame:
    info(f"{geo} - searching for prescriptions...")
    factors = costs.tolist() * n_days
    for i, f in enumerate(factors):
        factors[i] = f / limits[i] / n_days

    # get the input
    df = compute_context(geo, past_cases, population)
    past_actions = past_ips[-NB_LOOKBACK_DAYS:]
    past_context = df['PredictionRatio'].values[-NB_LOOKBACK_DAYS:]
    try:
        past_context = np.reshape(past_context, (NB_LOOKBACK_DAYS, 1))
    except ValueError:
        warning(f"{geo} - unable to determine past context")
        past_context = [[0] * len(NPI_LIMITS)] * NB_LOOKBACK_DAYS

    # generate a set of plans
    prescriptions_by_index = plan_generator.generate_plans(PRESCRIPTION_INDEXES,
                                                           PRESCRIPTION_CANDIDATES_PER_INDEX,
                                                           n_days,
                                                           factors,
                                                           limits)
    # evaluate the plans
    max_estimated_cases = 1e-6  # avoid divide by zero
    for candidate_prescriptions in prescriptions_by_index:
        for i, candidate_prescription in enumerate(candidate_prescriptions):
            candidate_prescription.estimated_cases = quantized_predictor.predict_geo(
                df,
                n_days,
                past_context,
                past_actions,
                np.reshape(candidate_prescription.actions, (n_days, 12)))
            max_estimated_cases = max(max_estimated_cases, candidate_prescription.estimated_cases)
            if i % 50 == 49:  # only log 1 of 50 plans, to reduce logging noise
                info(f"{geo} - {candidate_prescription.estimated_cases} {candidate_prescription.stringency}")

    # score the plans
    for prescription_index, plans in enumerate(prescriptions_by_index):
        # for low prescription indexes, we optimize for low stringency
        # otherwise, we favor lower cases
        stringency_weight = 2. * float(PRESCRIPTION_INDEXES - prescription_index) / float(PRESCRIPTION_INDEXES)
        for plan in plans:
            plan.score = (plan.estimated_cases / max_estimated_cases) * (stringency_weight * plan.stringency / 12.)

    # find the best prescriptions for each prescription index
    best_prescriptions = []
    for i, candidate_prescriptions in enumerate(prescriptions_by_index):
        best = None
        for candidate_prescription in candidate_prescriptions:
            if best is None:
                best = candidate_prescription
            elif candidate_prescription.score < best.score:
                best = candidate_prescription
            elif candidate_prescription.score == best.score and candidate_prescription.estimated_cases < best.estimated_cases:
                best = candidate_prescription
        info(f"{geo} - best index {i} - {best.estimated_cases} {best.stringency}")
        best_prescriptions.append(best)

    # generate the dataframe we'll return
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
    df_prescriptions = pd.DataFrame(data=data, columns=['PrescriptionIndex',
                                                        'CountryName',
                                                        'RegionName',
                                                        'Date',
                                                        C1, C2, C3, C4, C5, C6, C7, C8, H1, H2, H3, H6,
                                                        'EstimatedCases',
                                                        'EstimatedScore'])
    return df_prescriptions


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
