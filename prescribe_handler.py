import logging
import multiprocessing
import os
import time
from datetime import datetime
from logging import info

import numpy as np
import pandas as pd

from covid_xprize.examples.prescriptors.neat.utils import load_ips_file, add_geo_id, CASES_COL, IP_COLS, PRED_CASES_COL, \
    get_predictions, prepare_historical_df
from covid_xprize.standard_predictor.xprize_predictor import ADDITIONAL_BRAZIL_CONTEXT, ADDITIONAL_UK_CONTEXT, \
    US_PREFIX, ADDITIONAL_US_STATES_CONTEXT, ADDITIONAL_CONTEXT_FILE
from pandora.prescription_generator import PrescriptionGenerator
from pandora.quantized_constants import NPI_LIMITS, THREADS, PRESCRIPTION_CANDIDATES_PER_INDEX_RUN_1, \
    PRESCRIPTION_CANDIDATES_PER_INDEX_RUN_2
from prescribe_handler_process import prescribe_loop_for_geo

if not os.path.exists('logs'):
    os.makedirs('logs')
logging.basicConfig(level=logging.INFO, format='%(asctime)s\t%(levelname)s\t%(filename)s\t%(message)s')
logging.getLogger().handlers = [logging.FileHandler(f"logs/prescribe-{time.strftime('%Y-%m-%d')}"),
                                logging.StreamHandler()]


def prescribe(start_date_str: str,
              end_date_str: str,
              path_to_prior_ips_file: str,
              path_to_cost_file: str,
              output_file_path) -> None:
    info(f"prescription started @ {datetime.now()}")
    info(f"prescription from {start_date_str} to {end_date_str}")
    info(f"prescription with past IPS:   {path_to_prior_ips_file}")
    info(f"prescription with past costs: {path_to_cost_file}")
    info(f"prescription with output to:  {output_file_path}")

    start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d')
    end_date = pd.to_datetime(end_date_str, format='%Y-%m-%d')
    n_days = (end_date - start_date).days + 1

    # Load the past IPs data
    past_ips_df = load_ips_file(path_to_prior_ips_file)
    geos = past_ips_df['GeoID'].unique()

    # Load historical data with basic preprocessing
    df = prepare_historical_df()

    # Restrict it to dates before the start_date
    df = df[df['Date'] <= start_date]

    # Create past case data arrays for all geos
    past_cases = {}
    for geo in geos:
        geo_df = df[df['GeoID'] == geo]
        past_cases[geo] = np.maximum(0, np.array(geo_df[CASES_COL]))

    # Create past ip data arrays for all geos
    past_ips = {}
    for geo in geos:
        geo_df = past_ips_df[past_ips_df['GeoID'] == geo]
        past_ips[geo] = np.array(geo_df[IP_COLS])

    # Fill in any missing case data before start_date
    # using predictor given past_ips_df.
    # Note that the following assumes that the df returned by prepare_historical_df()
    # has the same final date for all regions. This has been true so far, but relies
    # on it being true for the Oxford data csv loaded by prepare_historical_df().
    last_historical_data_date_str = df['Date'].max()
    last_historical_data_date = pd.to_datetime(last_historical_data_date_str,
                                               format='%Y-%m-%d')
    if last_historical_data_date + pd.Timedelta(days=1) < start_date:
        info("Filling in missing data...")
        missing_data_start_date = last_historical_data_date + pd.Timedelta(days=1)
        missing_data_start_date_str = datetime.strftime(missing_data_start_date, format='%Y-%m-%d')
        missing_data_end_date = start_date - pd.Timedelta(days=1)
        missing_data_end_date_str = datetime.strftime(missing_data_end_date, format='%Y-%m-%d')
        pred_df = get_predictions(missing_data_start_date_str,
                                  missing_data_end_date_str,
                                  past_ips_df)
        pred_df = add_geo_id(pred_df)
        for geo in geos:
            geo_df = pred_df[pred_df['GeoID'] == geo].sort_values(by='Date')
            pred_cases_arr = np.array(geo_df[PRED_CASES_COL])
            past_cases[geo] = np.append(past_cases[geo], pred_cases_arr)
    else:
        info("No missing data.")

    # Load IP costs to condition prescriptions
    cost_df = pd.read_csv(path_to_cost_file)
    cost_df['RegionName'] = cost_df['RegionName'].fillna("")
    cost_df = add_geo_id(cost_df)
    geo_costs = {}
    for geo in geos:
        costs = cost_df[cost_df['GeoID'] == geo]
        cost_arr = np.array(costs[IP_COLS])[0]
        geo_costs[geo] = cost_arr

    # perform iterations while we have a time-budget
    info(f"initializing prescriptions generator")
    limits = NPI_LIMITS * n_days
    prescription_generator = PrescriptionGenerator(n_days, limits)

    info(f"prescription run 1 started @ {datetime.now()}")
    prescribe_loop(geos,
                   geo_costs,
                   past_cases,
                   past_ips,
                   n_days,
                   output_file_path,
                   start_date,
                   PRESCRIPTION_CANDIDATES_PER_INDEX_RUN_1,
                   prescription_generator,
                   limits)

    info(f"prescription run 2 started @ {datetime.now()}")
    prescribe_loop(geos,
                   geo_costs,
                   past_cases,
                   past_ips,
                   n_days,
                   output_file_path,
                   start_date,
                   PRESCRIPTION_CANDIDATES_PER_INDEX_RUN_2,
                   prescription_generator,
                   limits)
    info(f"prescription run 2 ended @ {datetime.now()}")


def prescribe_loop(geos,
                   geo_costs,
                   past_cases,
                   past_ips,
                   n_days: int,
                   output_file_path: str,
                   start_date,
                   prescription_candidates_per_index: int,
                   prescription_generator: PrescriptionGenerator,
                   limits):
    jobs = []
    populations = load_populations(geos)
    for geo in geos:
        arguments = [geo,
                     start_date,
                     geo_costs[geo],
                     past_cases[geo],
                     past_ips[geo],
                     n_days,
                     limits,
                     populations[geo],
                     prescription_candidates_per_index,
                     prescription_generator]
        jobs.append(arguments)

    # execute the mp pool
    with multiprocessing.Pool(processes=THREADS) as pool:
        results = pool.starmap(prescribe_loop_for_geo, jobs)
        df_output = pd.concat(results)

    # write the data
    write(df_output, output_file_path)


def write(df: pd.DataFrame, path: str):
    info('writing output!')
    output_dir = os.path.dirname(path)
    if output_dir != '':
        os.makedirs(output_dir, exist_ok=True)
    df = df.sort_values(['PrescriptionIndex', 'CountryName', 'RegionName', 'Date'])
    df.to_csv(path, index=False)


def load_populations(geos):
    df = load_additional_context_df()
    populations = {}
    for geo in geos:
        populations[geo] = df.loc[df['GeoID'] == geo]['Population'].max()
    return populations


def load_additional_context_df():
    # File containing the population for each country
    additional_context_df = pd.read_csv(ADDITIONAL_CONTEXT_FILE, usecols=['CountryName', 'Population'])
    additional_context_df['GeoID'] = additional_context_df['CountryName'] + '__'

    # US states population
    additional_us_states_df = pd.read_csv(ADDITIONAL_US_STATES_CONTEXT, usecols=['NAME', 'POPESTIMATE2019'])
    additional_us_states_df.rename(columns={'POPESTIMATE2019': 'Population'}, inplace=True)
    additional_us_states_df['GeoID'] = US_PREFIX + additional_us_states_df['NAME']
    additional_context_df = additional_context_df.append(additional_us_states_df)

    # UK population
    additional_uk_df = pd.read_csv(ADDITIONAL_UK_CONTEXT)
    additional_context_df = additional_context_df.append(additional_uk_df)

    # Brazil population
    additional_brazil_df = pd.read_csv(ADDITIONAL_BRAZIL_CONTEXT)
    additional_context_df = additional_context_df.append(additional_brazil_df)

    additional_context_df['GeoID'] = additional_context_df['GeoID'].apply(lambda x: x.replace(' / ', '__'))
    return additional_context_df
