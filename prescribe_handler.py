import logging
import os
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime
from logging import info

from covid_xprize.examples.prescriptors.neat.utils import load_ips_file, add_geo_id, CASES_COL, IP_COLS, PRED_CASES_COL, \
    get_predictions, prepare_historical_df


def prescribe(start_date_str: str,
              end_date_str: str,
              path_to_prior_ips_file: str,
              path_to_cost_file: str,
              output_file_path) -> None:
    """
    Generates and saves a file with daily intervention plan prescriptions for the given countries, regions and prior
    intervention plans, between start_date and end_date, included.
    :param start_date: day from which to start making prescriptions, as a string, format YYYY-MM-DDD
    :param end_date: day on which to stop making prescriptions, as a string, format YYYY-MM-DDD
    :param path_to_prior_ips_file: path to a csv file containing the intervention plans between inception date
    (Jan 1 2020) and end_date, for the countries and regions for which a prescription is needed
    :param path_to_cost_file: path to a csv file containing the cost of each individual intervention, per country
    See covid_xprize/validation/data/uniform_random_costs.csv for an example
    :param output_file_path: path to file to save the prescriptions to
    :return: Nothing. Saves the generated prescriptions to an output_file_path csv file
    See 2020-08-01_2020-08-04_prescriptions_example.csv for an example
    """
    info(
        f"prescribing [{start_date_str}-{end_date_str}] [{path_to_prior_ips_file}] [{path_to_cost_file}] [{output_file_path}]")

    start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d')
    end_date = pd.to_datetime(end_date_str, format='%Y-%m-%d')

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


    print(geo_costs)


def prescribe_iteration(geos):
    # split by country / region
    pass


def initialize():
    LOG_FORMAT = '%(asctime)s\t%(levelname)s\t%(filename)s\t%(message)s'
    for handler in logging.getLogger().handlers:
        logging.getLogger().removeHandler(handler)

    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.getLogger().setLevel(logging.INFO)
    logFormatter = logging.Formatter(LOG_FORMAT)
    rootLogger = logging.getLogger()

    if not os.path.exists('logs'):
        os.makedirs('logs')
    date_str = time.strftime("%Y_%m_%d")
    fileHandler = logging.FileHandler("{0}/{1}.log".format('./logs', f"prescribe-{date_str}"))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)


initialize()
