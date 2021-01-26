import logging
import os
import sys
import time
from logging import info


def prescribe(start_date: str,
              end_date: str,
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
    info(f"prescribing [{start_date}-{end_date}] [{path_to_prior_ips_file}] [{path_to_cost_file}] [{output_file_path}]")


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
