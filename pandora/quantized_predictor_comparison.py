import os
from datetime import datetime

import covid_xprize.standard_predictor.xprize_predictor as xprize_predictor
import quantized_predictor

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_INPUT_FILE = os.path.join(ROOT_DIR, "../covid_xprize", "validation/data/future_ip.csv")

# test with ~90 days
START_DATE = "2020-08-01"
END_DATE = "2020-11-01"

standard_predictor = xprize_predictor.XPrizePredictor()
quantized_predictor = quantized_predictor.QuantizedPredictor()

print(f"{datetime.now()} - predicting with quantized predictor")
quantized_pred_df = quantized_predictor.predict(START_DATE, END_DATE, EXAMPLE_INPUT_FILE)
print(quantized_pred_df['PredictedDailyNewCases'].sum())

print(f"{datetime.now()} - predicting with standard predictor")
standard_pred_df = standard_predictor.predict(START_DATE, END_DATE, EXAMPLE_INPUT_FILE)
print(standard_pred_df['PredictedDailyNewCases'].sum())

print(f"{datetime.now()} - finished")


