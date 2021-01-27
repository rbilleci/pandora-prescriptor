import logging
import time
import os
from logging import info

import pandas as pd

if not os.path.exists('logs'):
    os.makedirs('logs')
logging.basicConfig(level=logging.INFO, format='%(asctime)s\t%(levelname)s\t%(filename)s\t%(message)s')
logging.getLogger().handlers = [logging.FileHandler(f"logs/prescribe-{time.strftime('%Y-%m-%d')}"),
                                logging.StreamHandler()]


def prescribe_loop_for_geo(geo,
                           costs,
                           past_cases: pd.DataFrame,
                           past_ips: pd.DataFrame):
    # info(geo)
    # info(costs)
    info(type(past_ips))
    info(type(past_cases))
    # info(past_ips)
    # info(past_cases)
    return geo, costs
