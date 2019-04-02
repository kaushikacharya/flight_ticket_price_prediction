#!/usr/bin/env python

from datetime import datetime, timedelta
import json
from math import sqrt
import numpy as np
import os
import pandas as pd
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def compute_duration(duration_str, time_format='hour'):
    """Compute duration in minutes from the duration string

        Parameters
        ----------
        duration_str : str
                e.g. 2h 20m  It can also contain only hour or minutes.
        time_format : str
                Either 'minute' or 'hour'
    """
    assert time_format in ['hour', 'minute'], "Expected time_format should be either 'hour' or 'minute'"
    duration = 0
    hour_arr = re.findall(r'\d+h', duration_str)
    if len(hour_arr) > 0:
        if time_format == "minute":
            duration += 60*int(hour_arr[0][:-1])
        elif time_format == "hour":
            duration += int(hour_arr[0][:-1])
        else:
            assert False, "Unexpected time_format: {0}".format(time_format)

    minute_arr = re.findall(r'\d+m', duration_str)
    if len(minute_arr) > 0:
        if time_format == "minute":
            duration += int(minute_arr[0][:-1])
        elif time_format == "hour":
            duration += int(minute_arr[0][:-1])*1.0/60
        else:
            assert False, "Unexpected time_format: {0}".format(time_format)

    return duration


class FlightTicketPrice:
    def __init__(self, data_folder, train_size, random_state=100):
        self.data_folder = data_folder
        self.train_df = None
        self.test_df = None

        self.train_size = train_size
        self.random_state = random_state

        self.min_date_of_journey = None
        self.one_hot_encoder = None

        self.feature_train_matrix = None
        self.feature_validation_matrix = None
        self.feature_test_matrix = None

        self.train_index_arr = None
        self.validation_index_arr = None

        self.clf_random_forest_regression = None

        pd.set_option('display.width', 5000)
        pd.set_option('display.max_columns', 60)

    def load_train_data(self, train_data_file, verbose=False):
        train_data_file_path = os.path.join(self.data_folder, train_data_file)
        self.train_df = pd.read_excel(io=train_data_file_path)

        nan_train_df = self.train_df[self.train_df.isnull().any(1)]
        if len(nan_train_df) > 0:
            print("Dropping index: {0} from train_df".format(nan_train_df.index))
            self.train_df = self.train_df.drop(nan_train_df.index, axis=0).reset_index(drop=True)

        if verbose:
            print("Train data:")
            print("columns: {0}".format(self.train_df.columns))
            print("column data types:\n{0}".format(self.train_df.dtypes))
            # self.train_df = self.train_df.infer_objects()
            # print("column data types: {0}".format(self.train_df.dtypes))

    def load_test_data(self, test_data_file_name):
        test_data_file_path = os.path.join(self.data_folder, test_data_file_name)
        self.test_df = pd.read_excel(io=test_data_file_path)

    @staticmethod
    def preprocess_data(df):
        # Combine columns: Date_of_Journey, Dep_Time
        df['Departure_datetime'] = pd.to_datetime(df['Date_of_Journey'] + " " + df['Dep_Time'], format='%d/%m/%Y %H:%M')
        # TODO Though this makes the data type of Departure_datetime as datetime, but its good to have a check.
        df['Date_of_Journey'] = pd.to_datetime(df['Date_of_Journey'], format='%d/%m/%Y')
        # https://stackoverflow.com/questions/39207640/python-pandas-how-do-i-convert-from-datetime64ns-to-datetime
        df['Weekday'] = df['Departure_datetime'].dt.strftime('%a')

        # Create source, destination airport code from the route.
        #   This will ensure that Delhi and New Delhi are treated same as their airport codes are same.
        df['Source_code'] = df.apply(
            lambda row: row['Route'].split()[0] if isinstance(row['Route'], unicode) else np.nan, axis=1)
        df['Destination_code'] = df.apply(
            lambda row: row['Route'].split()[-1] if isinstance(row['Route'], unicode) else np.nan, axis=1)

        # Create a numeric column for Total_Stops
        df['n_stops'] = df.apply(lambda row: (
        0 if len(row['Total_Stops'].split()) == 1 else int(row['Total_Stops'].split()[0])) if isinstance(
            row['Total_Stops'], unicode) else np.nan, axis=1)
        df['n_stops'] = df.apply(lambda row: (
        0 if len(row['Total_Stops'].split()) == 1 else int(row['Total_Stops'].split()[0])) if isinstance(
            row['Total_Stops'], unicode) else np.nan, axis=1)

        df['Duration_numeric'] = df.apply(lambda row: compute_duration(duration_str=row['Duration']), axis=1)

        # Changes based on observation of data
        df.loc[(df['Airline'] == 'Jet Airways Business') & (df['Additional_Info'] == 'No info'), 'Additional_Info'] = 'Business class'
        df.loc[(df['Airline'] == 'Jet Airways Business'), 'Airline'] = 'Jet Airways'

        return df

    def set_min_date_of_journey(self):
        # TODO Check that column 'Date_of_Journey' has been converted to datetime
        self.min_date_of_journey = self.train_df['Date_of_Journey'].min()

    @staticmethod
    def get_subset_data(df, filter_dict):
        index_series = pd.Series(np.full(shape=len(df), fill_value=True, dtype=bool))
        for i, key in enumerate(filter_dict.keys()):
            index_series &= (df[key] == filter_dict[key])

        df_subset = df[index_series]
        return df_subset

    def analyse(self, verbose=True):
        print("Airline: {0}".format(self.train_df['Airline'].unique()))

        y_true_arr = []
        y_predicted_arr = []

        for airline in self.train_df['Airline'].unique():
            if airline.startswith('Multiple'):
                continue
            source_codes_airline = self.train_df[self.train_df['Airline'] == airline]['Source_code'].unique()
            destination_codes_airline = self.train_df[self.train_df['Airline'] == airline]['Destination_code'].unique()

            for source_code in source_codes_airline:
                for destination_code in destination_codes_airline:
                    print("\nsource code: {0} :: destination code: {1}".format(source_code, destination_code))
                    filter_dict = {'Airline': airline, 'Source_code': source_code, 'Destination_code': destination_code, 'n_stops': 0}
                    df = self.get_subset_data(df=self.train_df, filter_dict=filter_dict)
                    if len(df) == 0:
                        continue
                    y_true_cur_source_destination = []
                    y_predicted_cur_source_destination = []
                    # reset index
                    # df.reset_index(drop=True, inplace=True)
                    # Predicting for each of the train sample based on train data with all but the current one
                    for row_index in df.index:
                        cur_departure_datetime = df.loc[row_index, "Departure_datetime"]
                        df_dropped_cur_row = df.drop(row_index, axis=0).reset_index(drop=True)
                        true_price = df.loc[row_index, "Price"]
                        predicted_price = self.predict_price_using_neighbors(df=df_dropped_cur_row, departure_datetime=cur_departure_datetime)
                        if verbose:
                            print("\tDeparture datetime: {0} :: Actual Price: {1} :: Predicted Price: {2}".format(
                                df.loc[row_index, "Departure_datetime"], true_price, predicted_price))

                        y_true_cur_source_destination.append(true_price)
                        y_predicted_cur_source_destination.append(predicted_price)

                    # RMSE for current source -> destination
                    try:
                        rmse_cur_source_destination = sqrt(mean_squared_log_error(y_true=y_true_cur_source_destination, y_pred=y_predicted_cur_source_destination))
                    except Exception, err:
                        rmse_cur_source_destination = np.nan
                    print("\n\tRMSE for current source->destination: {0} :: n_samples: {1}".format(rmse_cur_source_destination, len(df.index)))

                    y_true_arr.extend(y_true_cur_source_destination)
                    y_predicted_arr.extend(y_predicted_cur_source_destination)

        try:
            rmse_train = sqrt(mean_squared_log_error(y_true=y_true_arr, y_pred=y_predicted_arr))
        except Exception, err:
            print("Nan count: {0}".format(len(np.where(np.isnan(y_predicted_arr))[0])))
            y_true_arr = [y_true_arr[i] for i in range(len(y_true_arr)) if not np.isnan(y_predicted_arr[i])]
            y_predicted_arr = [y_predicted_arr[i] for i in range(len(y_predicted_arr)) if not np.isnan(y_predicted_arr[i])]
            rmse_train = sqrt(mean_squared_log_error(y_true=y_true_arr, y_pred=y_predicted_arr))
        print("\nRMSE (train): {0} :: n_samples: {1}".format(rmse_train, len(y_true_arr)))

    def route_statistics(self):
        """Route Statistics

            Assumption: Pre-processing of both train and test data frames has been done
        """
        # Combine train and test data frames
        df = pd.concat([self.train_df, self.test_df], ignore_index=True)

        # columns for route statistics: Airline, Source_code, Destination_code, Route, n_stops
        df_route_stats = pd.DataFrame(columns=['Airline', 'Source_code', 'Destination_code', 'Route', 'n_stops', 'count'])
        # https://stackoverflow.com/questions/19384532/how-to-count-number-of-rows-per-group-and-other-statistics-in-pandas-group-by
        df_route_stats = df.groupby(by=['Airline', 'Source_code', 'Destination_code', 'Route', 'n_stops']).size().reset_index(name='count')
        print("route stats:\n{0}".format(df_route_stats))
        '''
        for airline in df['Airline'].unique():
            print("\n-----------------------------------------\nAirline: {0}".format(airline))
            source_codes_airline = df[df['Airline'] == airline]['Source_code'].unique()
            destination_codes_airline = df[df['Airline'] == airline]['Destination_code'].unique()

            for source_code in source_codes_airline:
                for destination_code in destination_codes_airline:
                    print("\nsource code: {0} :: destination code: {1}".format(source_code, destination_code))
                    df[(df['Airline']==airline) & (df['Source_code']==source_code) & (df['Destination_code']==destination_code)]
        '''

    def neighbor_distance_statistics(self):
        """Collecting statistics regarding distance between neighbors

            Note
            ----
            Assumption: train_df is in sorted order of Departure_datetime
        """
        # Compute
        #   1. Distance between different timings of the same day on the same route for each airline.
        #   2. Distance between airlines for close by timings.
        #   3. Distance between neighboring dates for same timing.
        price_column = self.train_df.columns.get_loc('Price')
        departure_datetime_column = self.train_df.columns.get_loc('Departure_datetime')
        dep_time_column = self.train_df.columns.get_loc('Dep_Time')
        date_of_journey_column = self.train_df.columns.get_loc('Date_of_Journey')

        def update_price_ratio_dict(dep_time_to_price_dict, same_day_price_ratio_dict):
            if len(dep_time_to_price_dict) > 1:
                dep_time_sorted = sorted(dep_time_to_price_dict.keys())
                for key_i in range(len(dep_time_sorted) - 1):
                    for key_j in range(key_i + 1, len(dep_time_sorted)):
                        dep_time_first = dep_time_sorted[key_i]
                        dep_time_second = dep_time_sorted[key_j]

                        if dep_time_first not in same_day_price_ratio_dict:
                            same_day_price_ratio_dict[dep_time_first] = dict()
                        if dep_time_second not in same_day_price_ratio_dict[dep_time_first]:
                            same_day_price_ratio_dict[dep_time_first][dep_time_second] = []

                        price_ratio = dep_time_to_price_dict[dep_time_second] * 1.0 / dep_time_to_price_dict[
                            dep_time_first]
                        same_day_price_ratio_dict[dep_time_first][dep_time_second].append(price_ratio)

        same_day_price_ratio_stats = dict()
        for airline in self.train_df['Airline'].unique():
            same_day_price_ratio_stats[airline] = dict()
            source_codes_airline = self.train_df[self.train_df['Airline'] == airline]['Source_code'].unique()

            for source_code in source_codes_airline:
                destination_codes_airline = self.train_df[(self.train_df['Airline'] == airline) & (self.train_df['Source_code'] == source_code)][
                    'Destination_code'].unique()
                same_day_price_ratio_stats[airline][source_code] = dict()
                for destination_code in destination_codes_airline:
                    same_day_price_ratio_stats[airline][source_code][destination_code] = dict()
                    n_stops_arr = sorted(self.train_df[(self.train_df['Airline'] == airline) &
                                                   (self.train_df['Source_code'] == source_code) &
                                                   (self.train_df['Destination_code'] == destination_code)]['n_stops'].unique())
                    # TODO Convert n_stops column into int. If not able to do why is it so?
                    n_stops_arr = [int(float(x)) for x in n_stops_arr]
                    for n_stops in n_stops_arr:
                        same_day_price_ratio_stats[airline][source_code][destination_code][n_stops] = dict()
                        routes_arr = self.train_df[(self.train_df['Airline'] == airline) &
                                                   (self.train_df['Source_code'] == source_code) &
                                                   (self.train_df['Destination_code'] == destination_code) &
                                                   (self.train_df['n_stops'] == n_stops)]['Route'].unique()
                        for route in routes_arr:
                            same_day_price_ratio_stats[airline][source_code][destination_code][n_stops][route] = dict()
                            same_day_price_ratio_dict = dict()
                            filter_dict = {'Airline': airline, 'Source_code': source_code, 'Destination_code': destination_code,
                                           'n_stops': n_stops, 'Route': route}
                            df = self.get_subset_data(df=self.train_df, filter_dict=filter_dict)

                            i = 0
                            prev_date_of_journey = df.iloc[i, date_of_journey_column]
                            prev_dep_time = df.iloc[i, dep_time_column]
                            dep_time_to_price_dict = dict()
                            dep_time_to_price_dict[prev_dep_time] = df.iloc[i, price_column]
                            i += 1
                            while i < len(df):
                                cur_date_of_journey = df.iloc[i, date_of_journey_column]
                                cur_dep_time = df.iloc[i, dep_time_column]
                                cur_price = df.iloc[i, price_column]

                                if prev_date_of_journey != cur_date_of_journey:
                                    update_price_ratio_dict(dep_time_to_price_dict, same_day_price_ratio_dict)

                                    # reset dep time to price mapping
                                    dep_time_to_price_dict = dict()
                                else:
                                    if cur_dep_time not in dep_time_to_price_dict:
                                        dep_time_to_price_dict[cur_dep_time] = cur_price
                                    else:
                                        # If multiple instances of price available, then consider the lowest one
                                        if cur_price < dep_time_to_price_dict[cur_dep_time]:
                                            dep_time_to_price_dict[cur_dep_time] = cur_price

                                i += 1
                                prev_dep_time = cur_dep_time
                                prev_date_of_journey = cur_date_of_journey

                            update_price_ratio_dict(dep_time_to_price_dict, same_day_price_ratio_dict)
                            same_day_price_ratio_stats[airline][source_code][destination_code][n_stops][route] = same_day_price_ratio_dict

        output_folder = "../statistics"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        with open(os.path.join(output_folder, "same_day_price_ratio_stats.json"), "w") as fd:
            json.dump(obj=same_day_price_ratio_stats, fp=fd)

    def load_neighbor_distance_statistics(self):
        stats_folder = "../statistics"
        with open(os.path.join(stats_folder, "same_day_price_ratio_stats.json")) as fd:
            same_day_price_ratio_stats = json.load(fp=fd)

        return same_day_price_ratio_stats

    def predict_price_using_neighbors(self, df, departure_datetime):
        """Predict flight ticket price

            Note
            ----
            df : subset of data frame matching the condition.
                Later we can allow to search for closer condition if the condition doesn't match exactly.
                Also df index reset has been done by calling function before calling this function.
        """
        predicted_price = np.nan
        if len(df) == 0:
            return predicted_price

        series_post_cur_departure = df['Departure_datetime'] > departure_datetime
        df_post_cur_departure = df[df['Departure_datetime'] > departure_datetime]

        price_column_location = df.columns.get_loc('Price')
        if len(df_post_cur_departure) == 0:
            predicted_price = df.iloc[len(df)-1, price_column_location]
        else:
            post_cur_index = df_post_cur_departure.index[0]
            if post_cur_index == 0:
                predicted_price = df.iloc[0, price_column_location]
            else:
                predicted_price = (df.iloc[post_cur_index-1, price_column_location] + df.iloc[post_cur_index, price_column_location])/2

        return predicted_price

    def analyse_test_data(self):
        same_day_price_ratio_stats = self.load_neighbor_distance_statistics()

        for i in range(len(self.test_df)):
            index_row = self.test_df.index[i]
            airline = self.test_df.loc[index_row, 'Airline']
            source_code = self.test_df.loc[index_row, 'Source_code']
            destination_code = self.test_df.loc[index_row, 'Destination_code']
            n_stops = str(self.test_df.loc[index_row, 'n_stops'])
            route = self.test_df.loc[index_row, 'Route']
            date_of_journey = self.test_df.loc[index_row, 'Date_of_Journey']

            if airline not in same_day_price_ratio_stats:
                print("Airline: {0} missing in price ratio stats for i: {1} :: index: {2}".format(airline, i, index_row))
                continue

            if source_code not in same_day_price_ratio_stats[airline]:
                print("Source code: {0} missing in price ratio stats for airline: {1} :: i: {2} :: index: {3}".format(source_code, airline, i, index_row))
                continue

            if destination_code not in same_day_price_ratio_stats[airline][source_code]:
                print("Destination code: {0} missing in price ratio stats for airline: {1} :: source code: {2} :: i: {3} :: index: {4}".format(destination_code, airline, source_code, i, index_row))
                continue

            if n_stops not in same_day_price_ratio_stats[airline][source_code][destination_code]:
                print("n_stops: {0} missing in price ratio stats for airline: {1} :: source code: {2} ::  destination code: {3} :: i: {4} :: index: {5}".format(
                    n_stops, airline, source_code, destination_code, i, index_row))
                continue

            if route not in same_day_price_ratio_stats[airline][source_code][destination_code][n_stops]:
                print("route: {0} missing in price ratio stats for airline: {1} :: source code: {2} ::  destination code: {3} :: n_stops: {4} :: i: {5} :: index: {6}".format(
                    route.encode('utf-8'), airline, source_code, destination_code, n_stops, i, index_row))
                continue

            train_df_subset = self.train_df[(self.train_df['Date_of_Journey'] == date_of_journey) &
                                            (self.train_df['Airline'] == airline) &
                                            (self.train_df['Source_code'] == source_code) &
                                            (self.train_df['Destination_code'] == destination_code) &
                                            (self.train_df['Route'] == route)]

            if len(train_df_subset) == 0:
                print("None sample on same date as airline: {0} :: source code: {1} ::  destination code: {2} :: n_stops: {3} :: i: {4} :: index: {5} :: route: {6}".format(
                    airline, source_code, destination_code, n_stops, i, index_row, route.encode('utf-8')))
                continue

    def create_feature_matrix(self):
        # https://stackoverflow.com/questions/46549991/using-both-numeric-and-categorical-variables-to-fit-a-decision-tree-using-sklear
        # Apply 1-hot encoding on categorical features and horizontally stack with numerical features
        print("columns: {0}".format(self.train_df.columns))
        print("column data types:\n{0}".format(self.train_df.dtypes))

        if self.min_date_of_journey is None:
            self.set_min_date_of_journey()

        # re-order test_df
        self.test_df.sort_index(inplace=True)

        self.train_df['Journey_day_from_min'] = [x.days for x in (self.train_df['Date_of_Journey'] - self.min_date_of_journey)]
        self.test_df['Journey_day_from_min'] = [x.days for x in (self.test_df['Date_of_Journey'] - self.min_date_of_journey)]

        self.train_df['Departure_numeric'] = [x.seconds/60 for x in (self.train_df['Departure_datetime'] - self.train_df['Date_of_Journey'])]
        self.test_df['Departure_numeric'] = [x.seconds/60 for x in (self.test_df['Departure_datetime'] - self.test_df['Date_of_Journey'])]

        train_index_arr, validation_index_arr = train_test_split(self.train_df.index, train_size=self.train_size, random_state=self.random_state)

        self.train_index_arr = train_index_arr
        self.validation_index_arr = validation_index_arr

        # https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-categorical-features
        self.one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
        columns_one_hot = ['Airline', 'Source_code', 'Destination_code', 'Route', 'Weekday', 'Additional_Info']
        self.one_hot_encoder.fit(self.train_df.loc[train_index_arr, columns_one_hot].values)
        # print(enc.categories_)

        one_hot_train_matrix = self.one_hot_encoder.transform(self.train_df.loc[train_index_arr, columns_one_hot].values).toarray().astype(int)
        one_hot_validation_matrix = self.one_hot_encoder.transform(self.train_df.loc[validation_index_arr, columns_one_hot].values).toarray().astype(int)
        one_hot_test_matrix = self.one_hot_encoder.transform(self.test_df[columns_one_hot].values).toarray().astype(int)

        columns_numeric = ['n_stops', 'Journey_day_from_min', 'Departure_numeric', 'Duration_numeric']

        numeric_train_matrix = self.train_df.loc[train_index_arr, columns_numeric].values
        self.feature_train_matrix = np.hstack((one_hot_train_matrix, numeric_train_matrix))

        numeric_validation_matrix = self.train_df.loc[validation_index_arr, columns_numeric].values
        self.feature_validation_matrix = np.hstack((one_hot_validation_matrix, numeric_validation_matrix))

        numeric_test_matrix = self.test_df[columns_numeric].values
        self.feature_test_matrix = np.hstack((one_hot_test_matrix, numeric_test_matrix))

    def train_random_forest(self, write_output=False):
        self.clf_random_forest_regression = RandomForestRegressor(n_estimators=100)
        self.clf_random_forest_regression.fit(X=self.feature_train_matrix, y=self.train_df.loc[self.train_index_arr, 'Price'])

        # Now predict on validation set
        y_predicted_arr = self.clf_random_forest_regression.predict(X=self.feature_validation_matrix)
        y_true_arr = self.train_df.loc[self.validation_index_arr, 'Price']

        if write_output:
            output_folder = "../output"

            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            train_subset_df = self.train_df.loc[self.train_index_arr, ]
            # sort in order of Departure datetime for analysis convenience
            train_subset_df.sort_values(by=u'Departure_datetime', inplace=True)
            with open(os.path.join(output_folder, "train_subset.csv"), "w") as fd:
                train_subset_df.to_csv(fd, index=False, encoding="utf-8")

            validation_subset_df = self.train_df.loc[self.validation_index_arr, ]
            validation_subset_df['Predicted_price'] = y_predicted_arr
            validation_subset_df['Diff_price'] = y_true_arr - y_predicted_arr
            # sort in order of Departure datetime for analysis convenience
            validation_subset_df.sort_values(by=u'Departure_datetime', inplace=True)

            with open(os.path.join(output_folder, "validation_subset.csv"), "w") as fd:
                validation_subset_df.to_csv(fd, index=False, encoding="utf-8")

        try:
            rmse_train = sqrt(mean_squared_log_error(y_true=y_true_arr, y_pred=y_predicted_arr))
        except Exception, err:
            print("Nan count: {0}".format(len(np.where(np.isnan(y_predicted_arr))[0])))
            y_true_arr = [y_true_arr[i] for i in range(len(y_true_arr)) if not np.isnan(y_predicted_arr[i])]
            y_predicted_arr = [y_predicted_arr[i] for i in range(len(y_predicted_arr)) if not np.isnan(y_predicted_arr[i])]
            rmse_train = sqrt(mean_squared_log_error(y_true=y_true_arr, y_pred=y_predicted_arr))

        print("\nRMSE (train): {0} :: n_samples(train): {1} :: n_samples(validation): {2}".format(rmse_train, len(self.train_index_arr), len(self.validation_index_arr)))

    def predict_random_forest(self):
        y_predicted_arr = self.clf_random_forest_regression.predict(X=self.feature_test_matrix)

        result_folder = "../result"
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        df = pd.DataFrame(data=y_predicted_arr, columns=['Price'])

        result_filepath = os.path.join(result_folder, "flight_ticket_price_prediction.xlsx")
        excel_writer = pd.ExcelWriter(path=result_filepath)
        df.to_excel(excel_writer=excel_writer, index=False)
        excel_writer.save()
        excel_writer.close()

if __name__ == "__main__":
    folder_data = "../data/Flight_Ticket_Participant_Datasets-20190305T100527Z-001/Flight_Ticket_Participant_Datasets"
    flight_ticket_price_obj = FlightTicketPrice(data_folder=folder_data, train_size=0.99, random_state=100)
    flight_ticket_price_obj.load_train_data(train_data_file="Data_Train.xlsx", verbose=True)
    flight_ticket_price_obj.load_test_data(test_data_file_name="Test_set.xlsx")

    # Pre-process data
    flight_ticket_price_obj.train_df = flight_ticket_price_obj.preprocess_data(flight_ticket_price_obj.train_df)
    flight_ticket_price_obj.test_df = flight_ticket_price_obj.preprocess_data(flight_ticket_price_obj.test_df)

    # Sort data based on departure date time
    flight_ticket_price_obj.train_df.sort_values(by=u'Departure_datetime', inplace=True)
    flight_ticket_price_obj.test_df.sort_values(by=u'Departure_datetime', inplace=True)

    # flight_ticket_price_obj.neighbor_distance_statistics()
    # flight_ticket_price_obj.load_neighbor_distance_statistics()
    # flight_ticket_price_obj.analyse(verbose=False)
    # flight_ticket_price_obj.route_statistics()
    # flight_ticket_price_obj.analyse_test_data()

    flight_ticket_price_obj.create_feature_matrix()
    flight_ticket_price_obj.train_random_forest(write_output=True)

    flight_ticket_price_obj.predict_random_forest()

"""
Problem Statement:
    https://www.machinehack.com/course/predict-the-flight-ticket-price-hackathon/

References:
    https://stackoverflow.com/questions/26763344/convert-pandas-column-to-datetime
        - 'Date_of_Journey' column needed to be converted from object data type to datetime

    https://stackoverflow.com/questions/30265723/python-create-a-new-column-from-existing-columns
        - Haleemur Ali's answer

    https://stackoverflow.com/questions/441147/how-to-subtract-a-day-from-a-date

    https://stackoverflow.com/questions/52173161/getting-a-list-of-indices-where-pandas-boolean-series-is-true

    https://stackoverflow.com/questions/40660088/get-first-row-of-dataframe-in-python-pandas-based-on-criteria
        - Tgsmith61591's answer

Approaches:
    - Start with K nearest neighbor
    - Then attempt time series regression
    - Create features to apply algorithm like Decision Tree, Random Forest
        https://www.analyticsindiamag.com/hands-on-tutorial-how-to-use-decision-tree-regression-to-solve-machinehacks-new-data-science-hackathon/
        - Here they have applied for Doctor fees prediction

TODO
    - 'Additional_Info' is missing info for several cases.
        - Need to identify the airlines which has multiple rows with everything same except the price.
        - Would have to tag it programmatically.
        - For now consider the lowest price in cases like this.
    - Compute numeric value for 'Duration' column
        - For 1 or more stop flights, compute estimated_flight_time using average time taken by flights on this route.
            - Do we have enough data for this?
        - ?? Won't the Route, n_stops can be a good alternate to 'Duration'
    - Combine Date_of_Journey and Dep_Time into a new column with datetime data type. [DONE]
        - Then this new column can be used for sorting and visual analyzing.
    - Create source, destination airport code from Route. [DONE]
        - Required as both Delhi, New Delhi have same airport code.
    - UserWarning: Boolean Series key will be reindexed to match DataFrame index.
        https://stackoverflow.com/questions/41710789/boolean-series-key-will-be-reindexed-to-match-dataframe-index

"""
