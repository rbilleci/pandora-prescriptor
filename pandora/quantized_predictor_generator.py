import os

import tensorflow as tf

import covid_xprize.standard_predictor.xprize_predictor as xp
from covid_xprize.standard_predictor.xprize_predictor import MODEL_WEIGHTS_FILE, DATA_FILE_PATH

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_INPUT_FILE = os.path.join(ROOT_DIR, "../covid_xprize", "validation/data/2020-09-30_historical_ip.csv")

QUANTIZED_FILE = "quantized_model.tflite"


def quantize():
    standard_predictor = xp.XPrizePredictor(MODEL_WEIGHTS_FILE, DATA_FILE_PATH)
    print('quantizing')
    converter = tf.lite.TFLiteConverter.from_keras_model(standard_predictor.predictor)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    quantized_model = converter.convert()
    open(QUANTIZED_FILE, "wb").write(quantized_model)


quantize()
