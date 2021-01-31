import numpy as np
import pandas as pd
import tensorflow as tf

from covid_xprize.standard_predictor.xprize_predictor import WINDOW_SIZE
from pandora.quantized_constants import QUANTIZED_FILE


class QuantizedPredictor(object):

    def __init__(self):
        self.quantized_predictor = tf.lite.Interpreter(QUANTIZED_FILE)
        self.quantized_predictor.allocate_tensors()
        print('quantized model ready')

    def predict_geo(self,
                    df: pd.DataFrame,
                    number_of_days,
                    initial_context_input,
                    initial_action_input,
                    future_action_sequence):
        if len(df) == 0:
            # we don't have historical data for this geo: return zeroes
            pred_new_cases = [0] * number_of_days
        else:
            pred_new_cases = self._predict_geo(df,
                                               initial_context_input,
                                               initial_action_input,
                                               future_action_sequence)
        total = 0.
        for pred in pred_new_cases[-number_of_days:]:
            total += pred
        return total

    def _predict_geo(self,
                     df: pd.DataFrame,
                     initial_context_input,
                     initial_action_input,
                     future_action_sequence):
        df = df[df.ConfirmedCases.notnull()]
        # Predictions with passed npis
        # Get the predictions with the passed NPIs
        preds = self._roll_out_predictions(self.quantized_predictor,
                                           initial_context_input,
                                           initial_action_input,
                                           future_action_sequence)
        # Gather info to convert to total cases
        prev_confirmed_cases = np.array(df.ConfirmedCases)
        prev_new_cases = np.array(df.NewCases)
        initial_total_cases = prev_confirmed_cases[-1]
        pop_size = np.array(df.Population)[-1]  # Population size doesn't change over time
        # Compute predictor's forecast
        pred_new_cases = self._convert_ratios_to_total_cases(
            preds,
            WINDOW_SIZE,
            prev_new_cases,
            initial_total_cases,
            pop_size)

        return pred_new_cases

    # Function for performing roll outs into the future
    @staticmethod
    def _roll_out_predictions(predictor,
                              initial_context_input,
                              initial_action_input,
                              future_action_sequence):
        nb_roll_out_days = future_action_sequence.shape[0]
        pred_output = np.zeros(nb_roll_out_days)
        context_input = np.expand_dims(np.copy(initial_context_input), axis=0)
        action_input = np.expand_dims(np.copy(initial_action_input), axis=0)
        for d in range(nb_roll_out_days):
            action_input[:, :-1] = action_input[:, 1:]
            # Use the passed actions
            action_sequence = future_action_sequence[d]
            action_input[:, -1] = action_sequence

            """ RB: start of customization """
            input_details = predictor.get_input_details()
            output_details = predictor.get_output_details()
            predictor.set_tensor(input_details[0]['index'], np.array(action_input, dtype=np.float32))
            predictor.set_tensor(input_details[1]['index'], np.array(context_input, dtype=np.float32))
            predictor.invoke()
            pred = predictor.get_tensor(output_details[0]['index'])
            """ RB: end of customization"""

            pred_output[d] = pred
            context_input[:, :-1] = context_input[:, 1:]
            context_input[:, -1] = pred
        return pred_output

    def _convert_ratios_to_total_cases(self,
                                       ratios,
                                       window_size,
                                       prev_new_cases,
                                       initial_total_cases,
                                       pop_size):
        new_new_cases = []
        prev_new_cases_list = list(prev_new_cases)
        curr_total_cases = initial_total_cases
        for ratio in ratios:
            new_cases = self._convert_ratio_to_new_cases(ratio,
                                                         window_size,
                                                         prev_new_cases_list,
                                                         curr_total_cases / pop_size)
            # new_cases can't be negative!
            new_cases = max(0, new_cases)
            # Which means total cases can't go down
            curr_total_cases += new_cases
            # Update prev_new_cases_list for next iteration of the loop
            prev_new_cases_list.append(new_cases)
            new_new_cases.append(new_cases)
        return new_new_cases

    # Functions for converting predictions back to number of cases
    @staticmethod
    def _convert_ratio_to_new_cases(ratio,
                                    window_size,
                                    prev_new_cases_list,
                                    prev_pct_infected):
        return (ratio * (1 - prev_pct_infected) - 1) * \
               (window_size * np.mean(prev_new_cases_list[-window_size:])) \
               + prev_new_cases_list[-window_size]
