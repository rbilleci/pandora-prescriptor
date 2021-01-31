import logging
import multiprocessing
import os
import time
from datetime import datetime, timedelta
from logging import info

import numpy as np
import pandas as pd

from covid_xprize.examples.prescriptors.neat.utils import load_ips_file, add_geo_id, CASES_COL, IP_COLS, PRED_CASES_COL, \
    get_predictions, prepare_historical_df
from covid_xprize.standard_predictor.xprize_predictor import ADDITIONAL_BRAZIL_CONTEXT, ADDITIONAL_UK_CONTEXT, \
    US_PREFIX, ADDITIONAL_US_STATES_CONTEXT, ADDITIONAL_CONTEXT_FILE
from pandora.quantized_constants import NPI_LIMITS, C1, H6, C2, C3, C4, C5, C6, C7, C8, H1, H2, H3
from prescribe_handler_process import prescribe_loop_for_geo

THREADS = 2

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
    """
    Generates and saves a file with daily intervention plan prescriptions for the given countries, regions and prior
    intervention plans, between start_date and end_date, included.
    :param start_date_str: day from which to start making prescriptions, as a string, format YYYY-MM-DDD
    :param end_date_str: day on which to stop making prescriptions, as a string, format YYYY-MM-DDD
    :param path_to_prior_ips_file: path to a csv file containing the intervention plans between inception date
    (Jan 1 2020) and end_date, for the countries and regions for which a prescription is needed
    :param path_to_cost_file: path to a csv file containing the cost of each individual intervention, per country
    See covid_xprize/validation/data/uniform_random_costs.csv for an example
    :param output_file_path: path to file to save the prescriptions to
    :return: Nothing. Saves the generated prescriptions to an output_file_path csv file
    See 2020-08-01_2020-08-04_prescriptions_example.csv for an example
    """
    info(
        f"prescribe [{start_date_str}-{end_date_str}] [{path_to_prior_ips_file}] [{path_to_cost_file}] [{output_file_path}]")

    start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d')
    end_date = pd.to_datetime(end_date_str, format='%Y-%m-%d')
    n_days = (end_date - start_date).days + 1
    print(f"prescribing for {n_days} days")

    # Load the past IPs data
    print("Loading past IPs data...")
    past_ips_df = load_ips_file(path_to_prior_ips_file)
    geos = past_ips_df['GeoID'].unique()

    # Load historical data with basic preprocessing
    print("Loading historical data...")
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
        print("Filling in missing data...")
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
        print("No missing data.")

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
    prescribe_loop(geos,
                   geo_costs,
                   past_cases,
                   past_ips,
                   n_days,
                   output_file_path,
                   start_date,
                   end_date)


def prescribe_loop(geos,
                   geo_costs,
                   past_cases,
                   past_ips,
                   n_days: int,
                   output_file_path: str,
                   start_date,
                   end_date):
    jobs = []
    limits = NPI_LIMITS * n_days
    populations = load_populations(geos)
    for geo in geos:
        arguments = [geo,
                     start_date,
                     geo_costs[geo],
                     past_cases[geo],
                     past_ips[geo],
                     n_days,
                     limits,
                     populations[geo]]
        jobs.append(arguments)
    with multiprocessing.Pool(processes=THREADS) as pool:
        results = pool.starmap(prescribe_loop_for_geo, jobs)
        df_output = pd.concat(results)
    # Create the output directory if necessary.
    info('writing output!')
    output_dir = os.path.dirname(output_file_path)
    if output_dir != '':
        os.makedirs(output_dir, exist_ok=True)
    df_output = df_output.sort_values(['PrescriptionIndex',
                                       'CountryName',
                                       'RegionName',
                                       'Date'])
    df_output.to_csv(output_file_path, index=False)


def load_populations(geos):
    df = load_additional_context_df()
    populations = {}
    for geo in geos:
        populations[geo] = df.loc[df['GeoID'] == geo]['Population'].max()
    return populations


def load_additional_context_df():
    # File containing the population for each country
    # Note: this file contains only countries population, not regions
    additional_context_df = pd.read_csv(ADDITIONAL_CONTEXT_FILE, usecols=['CountryName', 'Population'])
    additional_context_df['GeoID'] = additional_context_df['CountryName'] + '__'

    # US states population
    additional_us_states_df = pd.read_csv(ADDITIONAL_US_STATES_CONTEXT, usecols=['NAME', 'POPESTIMATE2019'])
    # Rename the columns to match measures_df ones
    additional_us_states_df.rename(columns={'POPESTIMATE2019': 'Population'}, inplace=True)

    # Prefix with country name to match measures_df
    additional_us_states_df['GeoID'] = US_PREFIX + additional_us_states_df['NAME']

    # Append the new data to additional_df
    additional_context_df = additional_context_df.append(additional_us_states_df)

    # UK population
    additional_uk_df = pd.read_csv(ADDITIONAL_UK_CONTEXT)
    # Append the new data to additional_df
    additional_context_df = additional_context_df.append(additional_uk_df)

    # Brazil population
    additional_brazil_df = pd.read_csv(ADDITIONAL_BRAZIL_CONTEXT)
    # Append the new data to additional_df
    additional_context_df = additional_context_df.append(additional_brazil_df)

    additional_context_df['GeoID'] = additional_context_df['GeoID'].apply(lambda x: x.replace(' / ', '__'))
    return additional_context_df
