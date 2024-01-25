import pandas as pd
import math
import re
import datetime
from datetime import timedelta

# jewish holidays in feauture
JEWISH_HOLIDAYS = {
    "Pesach23": {"from": "2023-04-05", "to": "2023-04-12"},
    "Shavuot23": {"from": "2023-05-25", "to": "2023-05-26"},
    "Rosh Hashanah23": {"from": "2023-09-15", "to": "2023-09-17"},
    "Yom Kippur23": {"from": "2023-09-24", "to": "2023-09-25"},
    "Sukkot23": {"from": "2023-09-30", "to": "2023-10-05"},
    "Hanukkah23": {"from": "2023-12-07", "to": "2023-12-15"},
    "Pesach22": {"from": "2022-04-15", "to": "2022-04-22"},
    "Shavuot22": {"from": "2022-06-04", "to": "2022-06-05"},
    "Rosh Hashanah22": {"from": "2022-09-25", "to": "2022-09-27"},
    "Yom Kippur22": {"from": "2022-10-04", "to": "2022-10-05"},
    "Sukkot22": {"from": "2022-10-09", "to": "2022-10-16"},
    "Hanukkah22": {"from": "2022-12-18", "to": "2022-12-26"},
}
EXTENDED_HOLIDAYS = {}
for holiday, date_range in JEWISH_HOLIDAYS.items():
    from_date = datetime.datetime.strptime(date_range['from'], '%Y-%m-%d')
    to_date = datetime.datetime.strptime(date_range['to'], '%Y-%m-%d')

    extended_from_date = from_date - timedelta(days=3)
    extended_to_date = to_date + timedelta(days=3)

    EXTENDED_HOLIDAYS[holiday] = {"from": extended_from_date, "to": extended_to_date}


class FeatureCreator(object):
    def __init__(self,
                 data,
                 train_or_test,
                 siri_time_original_column_name = "nearest_siri_vehicle_location__recorded_at_time",
                 gtfs_time_original_column_name = "gtfs_ride_stop__arrival_time",
                 target_column_name = "scheduled_vs_real_time_difference_seconds"
                 ):
        if train_or_test not in ["train", "test"]:
            raise Exception("train_or_test must be either train or test")
        self.train_or_test = train_or_test
        self.data = data
        self.siri_time_original_column_name = siri_time_original_column_name
        self.gtfs_time_original_column_name = gtfs_time_original_column_name
        self.target_column_name = target_column_name

    def create_features(self):
        self.create_is_holiday_feature()
        self.create_population_features()
        self.create_weather_features()
        self.create_time_features()

        self.create_sequence_features()
        if self.train_or_test == "train":
            self.create_target_encoded_variable("gtfs_stop__city")
            self.create_target_encoded_variable("gtfs_route__operator_ref")
            # self.create_lagging_feature()
        else:
            self.create_target_encoded_variable_from_dict("gtfs_stop__city")
            self.create_target_encoded_variable_from_dict("gtfs_route__operator_ref")
            # self.create_lagging_feature()
        self.data["amount_cities_in_route"] = self.data.groupby("siri_ride_id")["gtfs_stop__city"].transform("nunique")

        return self.data

    def create_weather_features(self):
        # read from rain.csv
        rain_df = pd.read_csv('one_file_rain.csv')

        rain_df['date_to_join'] = pd.to_datetime(rain_df['תאריך'], format="%d/%m/%Y")
        rain_per_day = rain_df.groupby("date_to_join").count().reset_index()[["date_to_join", "תאריך"]].rename(
            columns={"תאריך": "rain_exists"})
        self.data["date_to_join"] = self.data["nearest_siri_vehicle_location__recorded_at_time"].dt.date
        self.data["date_to_join"] = pd.to_datetime(self.data["date_to_join"])
        self.data = pd.merge(self.data, rain_per_day, on="date_to_join", how="left")
        self.data["rain_exists"] = self.data["rain_exists"].apply(lambda x: True if x > 0 else False)

    def create_is_holiday_feature(self):
        def is_ride_in_holiday(ride, extended_holidays):
            for holiday in extended_holidays.values():
                if ride['nearest_siri_vehicle_location__recorded_at_time'] >= holiday['from'] and ride['nearest_siri_vehicle_location__recorded_at_time'] <= holiday['to']:
                    return True

            return False

        self.data['is_holiday'] = self.data.apply(lambda ride: is_ride_in_holiday(ride, EXTENDED_HOLIDAYS), axis=1)
        return True

    def create_population_features(self):

        # Assuming df is your DataFrame
        cities = self.data['gtfs_stop__city'].unique()

        # Initialize cities_dict with 0 as values for all cities
        cities_dict = {re.sub(r'[^א-ת]', '', city): 0 for city in cities}

        # Read data for city from population.csv
        population_df = pd.read_csv('population.csv')

        # Extract relevant columns from population_df
        city_name_column = population_df.columns[1]
        population_column = population_df.columns[7]

        # Keep only hebrew alphabet characters in the city name
        population_df[city_name_column] = population_df[city_name_column].apply(lambda x: re.sub(r'[^א-ת]', '', str(x)))

        # Run over all the cities in the df and add the score to the dict
        for city_orig in cities:
            city = re.sub(r'[^א-ת]', '', city_orig)
            try:
                city_population_str = \
                population_df.loc[population_df[city_name_column] == city, population_column].values[0]
                print(city_population_str + " " + city)
                # Convert the population from string to integer
                city_population = int(
                    city_population_str.replace(',', ''))  # assuming population values might have commas
                cities_dict[city] = city_population  # math.floor(city_population / 100000)
            except (ValueError, IndexError):
                # Handle cases where the population value is not a valid integer or city is not found
                print(f"Error processing city: {city}")

        # add the value of the city to the df
        self.data['city_population'] = self.data['gtfs_stop__city'].apply(lambda x: cities_dict[re.sub(r'[^א-ת]', '', str(x))])

    def create_time_features(self):
        self.data['weekday'] = self.data['nearest_siri_vehicle_location__recorded_at_time'].apply(
            lambda x: pd.to_datetime(x).weekday())
        self.data["hour_of_day"] = self.data["nearest_siri_vehicle_location__recorded_at_time"].dt.hour


    def create_target_encoded_variable_from_dict(self, column_name):
        # read csv file with the dictionary mapping of the target encoded variable
        target_encoded_dict = pd.read_csv("data/" + column_name + "_target_encoded.csv")
        # Merge the dictionary mapping of the target encoded variable with the data
        self.data = pd.merge(self.data, target_encoded_dict, on=column_name, how="left")
        # Fill nulls with 0
        self.data[column_name + "_target_encoded"] = self.data[column_name + "_target_encoded"].fillna(0)

    def create_target_encoded_variable(self, column_name):
        self.data[column_name + "_target_encoded"] = self.data.groupby(column_name)[self.target_column_name].transform('mean')
        # Fill nulls with 0
        self.data[column_name + "_target_encoded"] = self.data[column_name + "_target_encoded"].fillna(0)
        # Keep the dictionary mapping of the target encoded variable in an external csv file
        self.data[[column_name, column_name + "_target_encoded"]].drop_duplicates().to_csv("data/" + column_name + "_target_encoded.csv", index=False)

    def create_sequence_features(self):
        max_stop_sequences = self.data.groupby("gtfs_ride__gtfs_route_id").agg(
            {"gtfs_ride_stop__stop_sequence": "max"}).reset_index().rename(
                columns={"gtfs_ride_stop__stop_sequence": "max_stop_sequence"}
        )
        self.data = pd.merge(self.data, max_stop_sequences, on="gtfs_ride__gtfs_route_id")
        self.data["stop_sequence_ratio"] = (self.data["gtfs_ride_stop__stop_sequence"] / self.data["max_stop_sequence"]).round(2)
        self.data["stop_sequence"] = self.data["gtfs_ride_stop__stop_sequence"]

    def create_lagging_feature(self):
        # Make a window function taking the mean of the target variable for the previous 5 stops
        self.data = self.data.sort_values(by=['siri_ride_id', 'stop_sequence'])
        self.data["past_3_mean"] = self.data.groupby('siri_ride_id')['scheduled_vs_real_time_difference_seconds']\
            .rolling(window=3,
                     min_periods=1,
                     closed='left')\
            .mean()\
            .reset_index(level=0, drop=True)

        self.data["past_3_mean"] = self.data["past_3_mean"].fillna(0)


# training_data = pd.read_csv("data/dev_test_train_2022-05-01T00:00:00+03:00_2022-10-01T00:00:00+03:00_final_data.csv")
#
# training_data["nearest_siri_vehicle_location__recorded_at_time"] = pd.to_datetime(training_data["nearest_siri_vehicle_location__recorded_at_time"]).dt.tz_localize(None)
# training_data["gtfs_ride_stop__arrival_time"] = pd.to_datetime(training_data["gtfs_ride_stop__arrival_time"]).dt.tz_localize(None)
# training_data["scheduled_vs_real_time_difference"] = training_data["nearest_siri_vehicle_location__recorded_at_time"] - training_data["gtfs_ride_stop__arrival_time"]
# training_data["scheduled_vs_real_time_difference_seconds"] = training_data["scheduled_vs_real_time_difference"].dt.total_seconds()
#
#
# print(FeatureCreator(training_data).create_features())