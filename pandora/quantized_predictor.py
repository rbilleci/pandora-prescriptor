import numpy as np
import pandas as pd
import tensorflow as tf
from quantized_constants import QUANTIZED_FILE

from covid_xprize.standard_predictor.xprize_predictor import DATA_FILE_PATH, NPI_COLUMNS, \
    WINDOW_SIZE, NB_LOOKBACK_DAYS, NB_TEST_DAYS, CONTEXT_COLUMNS, ADDITIONAL_CONTEXT_FILE, ADDITIONAL_US_STATES_CONTEXT, \
    US_PREFIX, ADDITIONAL_UK_CONTEXT, ADDITIONAL_BRAZIL_CONTEXT


class QuantizedPredictor(object):

    def __init__(self, data_url=DATA_FILE_PATH):
        self.quantized_predictor = tf.lite.Interpreter(QUANTIZED_FILE)
        self.quantized_predictor.allocate_tensors()
        print('quantized model ready')

        # TODO: do we need this here?
        self.df = self._prepare_dataframe(data_url)

    def predict(self,
                start_date_str: str,
                end_date_str: str,
                path_to_ips_file: str) -> pd.DataFrame:
        # Load the npis into a DataFrame, handling regions
        npis_df = self.load_original_data(path_to_ips_file)
        x = self.predict_from_df(start_date_str, end_date_str, npis_df)
        return x

    def predict_from_df(self,
                        start_date_str: str,
                        end_date_str: str,
                        npis_df: pd.DataFrame) -> pd.DataFrame:
        start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d')
        end_date = pd.to_datetime(end_date_str, format='%Y-%m-%d')
        nb_days = (end_date - start_date).days + 1

        # Prepare the output
        forecast = {"CountryName": [],
                    "RegionName": [],
                    "Date": [],
                    "PredictedDailyNewCases": []}

        # Fix for past predictions
        geos = npis_df.GeoID.unique()
        truncated_df = self.df[self.df.Date < start_date_str]
        country_samples = self._create_country_samples(truncated_df, geos, False)

        # For each requested geo
        geos = npis_df.GeoID.unique()
        for g in geos:
            cdf = truncated_df[truncated_df.GeoID == g]
            if len(cdf) == 0:  # we don't have historical data for this geo: return zeroes
                pred_new_cases = [0] * nb_days
                geo_start_date = start_date
            else:  # Start predicting from start_date, unless there's a gap since last known date
                last_known_date = cdf.Date.max()
                geo_start_date = min(last_known_date + np.timedelta64(1, 'D'), start_date)
                # TODO: looks like this is the problem?
                npis_gdf = npis_df[(npis_df.Date >= geo_start_date) & (npis_df.Date <= end_date)]
                pred_new_cases = self._get_new_cases_preds(cdf, g, npis_gdf, country_samples)

            # Append forecast data to results to return
            country = npis_df[npis_df.GeoID == g].iloc[0].CountryName
            region = npis_df[npis_df.GeoID == g].iloc[0].RegionName
            for i, pred in enumerate(pred_new_cases):
                forecast["CountryName"].append(country)
                forecast["RegionName"].append(region)
                current_date = geo_start_date + pd.offsets.Day(i)
                forecast["Date"].append(current_date)
                forecast["PredictedDailyNewCases"].append(pred)

        forecast_df = pd.DataFrame.from_dict(forecast)
        # Return only the requested predictions
        return forecast_df[(forecast_df.Date >= start_date) & (forecast_df.Date <= end_date)]

    def predict_geo(self,
                    geo,
                    df_geo: pd.DataFrame,
                    df_geo_npis: pd.DataFrame,
                    number_of_days,
                    country_samples):
        if len(df_geo) == 0:
            # we don't have historical data for this geo: return zeroes
            pred_new_cases = [0] * number_of_days
        else:
            pred_new_cases = self._get_new_cases_preds(df_geo, geo, df_geo_npis, country_samples)
        total = 0.
        for pred in pred_new_cases:
            total += pred
        return total

    def _get_new_cases_preds(self, c_df, g, npis_df, country_samples):
        cdf = c_df[c_df.ConfirmedCases.notnull()]
        initial_context_input = country_samples[g]['X_test_context'][-1]
        initial_action_input = country_samples[g]['X_test_action'][-1]
        # Predictions with passed npis
        cnpis_df = npis_df[npis_df.GeoID == g]
        npis_sequence = np.array(cnpis_df[NPI_COLUMNS])
        # Get the predictions with the passed NPIs
        preds = self._roll_out_predictions(self.quantized_predictor,
                                           initial_context_input,
                                           initial_action_input,
                                           npis_sequence)
        # Gather info to convert to total cases
        prev_confirmed_cases = np.array(cdf.ConfirmedCases)
        prev_new_cases = np.array(cdf.NewCases)
        initial_total_cases = prev_confirmed_cases[-1]
        pop_size = np.array(cdf.Population)[-1]  # Population size doesn't change over time
        # Compute predictor's forecast
        pred_new_cases = self._convert_ratios_to_total_cases(
            preds,
            WINDOW_SIZE,
            prev_new_cases,
            initial_total_cases,
            pop_size)

        return pred_new_cases

    def _prepare_dataframe(self, data_url: str) -> pd.DataFrame:
        """
        Loads the Oxford dataset, cleans it up and prepares the necessary columns. Depending on options, also
        loads the Johns Hopkins dataset and merges that in.
        :param data_url: the url containing the original data
        :return: a Pandas DataFrame with the historical data
        """
        # Original df from Oxford
        df1 = self.load_original_data(data_url)

        # Additional context df (e.g Population for each country)
        df2 = self._load_additional_context_df()

        # Merge the 2 DataFrames
        df = df1.merge(df2, on=['GeoID'], how='left', suffixes=('', '_y'))

        # Drop countries with no population data
        df.dropna(subset=['Population'], inplace=True)

        #  Keep only needed columns
        columns = CONTEXT_COLUMNS + NPI_COLUMNS
        df = df[columns]

        # Fill in missing values
        self._fill_missing_values(df)

        # Compute number of new cases and deaths each day
        df['NewCases'] = df.groupby('GeoID').ConfirmedCases.diff().fillna(0)
        df['NewDeaths'] = df.groupby('GeoID').ConfirmedDeaths.diff().fillna(0)

        # Replace negative values (which do not make sense for these columns) with 0
        df['NewCases'] = df['NewCases'].clip(lower=0)
        df['NewDeaths'] = df['NewDeaths'].clip(lower=0)

        # Compute smoothed versions of new cases and deaths each day
        df['SmoothNewCases'] = df.groupby('GeoID')['NewCases'].rolling(
            WINDOW_SIZE, center=False).mean().fillna(0).reset_index(0, drop=True)
        df['SmoothNewDeaths'] = df.groupby('GeoID')['NewDeaths'].rolling(
            WINDOW_SIZE, center=False).mean().fillna(0).reset_index(0, drop=True)

        # Compute percent change in new cases and deaths each day
        df['CaseRatio'] = df.groupby('GeoID').SmoothNewCases.pct_change(
        ).fillna(0).replace(np.inf, 0) + 1
        df['DeathRatio'] = df.groupby('GeoID').SmoothNewDeaths.pct_change(
        ).fillna(0).replace(np.inf, 0) + 1

        # Add column for proportion of population infected
        df['ProportionInfected'] = df['ConfirmedCases'] / df['Population']

        # Create column of value to predict
        df['PredictionRatio'] = df['CaseRatio'] / (1 - df['ProportionInfected'])

        return df

    @staticmethod
    def load_original_data(data_url):
        latest_df = pd.read_csv(data_url,
                                parse_dates=['Date'],
                                encoding="ISO-8859-1",
                                dtype={"RegionName": str,
                                       "RegionCode": str},
                                error_bad_lines=False)
        # GeoID is CountryName / RegionName
        # np.where usage: if A then B else C
        latest_df["GeoID"] = np.where(latest_df["RegionName"].isnull(),
                                      latest_df["CountryName"],
                                      latest_df["CountryName"] + ' / ' + latest_df["RegionName"])
        return latest_df

    @staticmethod
    def _fill_missing_values(df):
        """
        # Fill missing values by interpolation, ffill, and filling NaNs
        :param df: Dataframe to be filled
        """
        df.update(df.groupby('GeoID').ConfirmedCases.apply(
            lambda group: group.interpolate(limit_area='inside')))
        # Drop country / regions for which no number of cases is available
        df.dropna(subset=['ConfirmedCases'], inplace=True)
        df.update(df.groupby('GeoID').ConfirmedDeaths.apply(
            lambda group: group.interpolate(limit_area='inside')))
        # Drop country / regions for which no number of deaths is available
        df.dropna(subset=['ConfirmedDeaths'], inplace=True)
        for npi_column in NPI_COLUMNS:
            df.update(df.groupby('GeoID')[npi_column].ffill().fillna(0))

    @staticmethod
    def _load_additional_context_df():
        # File containing the population for each country
        # Note: this file contains only countries population, not regions
        additional_context_df = pd.read_csv(ADDITIONAL_CONTEXT_FILE,
                                            usecols=['CountryName', 'Population'])
        additional_context_df['GeoID'] = additional_context_df['CountryName']

        # US states population
        additional_us_states_df = pd.read_csv(ADDITIONAL_US_STATES_CONTEXT,
                                              usecols=['NAME', 'POPESTIMATE2019'])
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

        return additional_context_df

    @staticmethod
    def _create_country_samples(df: pd.DataFrame, geos: list, is_training: bool) -> dict:
        """
        For each country, creates numpy arrays for Keras
        :param df: a Pandas DataFrame with historical data for countries (the "Oxford" dataset)
        :param geos: a list of geo names
        :param is_training: True if the data will be used for training, False if it's used for predicting
        :return: a dictionary of train and test sets, for each specified country
        """
        context_column = 'PredictionRatio'
        action_columns = NPI_COLUMNS
        """ RB
        outcome_column = 'PredictionRatio'
        """
        country_samples = {}
        for g in geos:
            cdf = df[df.GeoID == g]
            cdf = cdf[cdf.ConfirmedCases.notnull()]
            context_data = np.array(cdf[context_column])
            action_data = np.array(cdf[action_columns])
            context_samples = []
            action_samples = []
            if is_training:
                """ RB
                outcome_data = np.array(cdf[outcome_column])
                outcome_samples = []
                """
                nb_total_days = context_data.shape[0]
            else:
                nb_total_days = context_data.shape[0] + 1
            for d in range(NB_LOOKBACK_DAYS, nb_total_days):
                context_samples.append(context_data[d - NB_LOOKBACK_DAYS:d])
                action_samples.append(action_data[d - NB_LOOKBACK_DAYS:d])
                """ RB
                if is_training:
                    outcome_samples.append(outcome_data[d])
                """
            if len(context_samples) > 0:
                X_context = np.expand_dims(np.stack(context_samples, axis=0), axis=2)
                X_action = np.stack(action_samples, axis=0)
                country_samples[g] = {
                    'X_context': X_context,
                    'X_action': X_action,
                    'X_train_context': X_context[:-NB_TEST_DAYS],
                    'X_train_action': X_action[:-NB_TEST_DAYS],
                    'X_test_context': X_context[-NB_TEST_DAYS:],
                    'X_test_action': X_action[-NB_TEST_DAYS:],
                }
                """ RB
                if is_training:
                    y = np.stack(outcome_samples, axis=0)
                    country_samples[g]['y'] = y
                    country_samples[g]['y_train'] = y[:-NB_TEST_DAYS]
                    country_samples[g]['y_test'] = y[-NB_TEST_DAYS:]
                """
        return country_samples

    # Function for performing roll outs into the future
    @staticmethod
    def _roll_out_predictions(predictor, initial_context_input, initial_action_input, future_action_sequence):
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
            predictor.set_tensor(input_details[0]['index'], np.array(context_input, dtype=np.float32))
            predictor.set_tensor(input_details[1]['index'], np.array(action_input, dtype=np.float32))
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
