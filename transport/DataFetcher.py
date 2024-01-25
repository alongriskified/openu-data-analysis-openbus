import stride
import pandas as pd
import os
import datetime
from datetime import timedelta
import random

DATA_DIR = "data"
class DataFetcher(object):
    def __init__(self,
                 name,
                 lines,
                 start_date,
                 end_date,
                 time_period_per_sample_hop_in_days = 1,
                 amount_of_lines_to_sample_per_hop = 5,
                 limit_ratio_lines_to_gtfs_lines = 5,
                 amount_of_rides_to_sample_per_hop = 100,
                 limit_routes_per_single_day = 10000,
                 limit_stops_per_single_day = 100000
                 ):
        self.start_date = start_date
        self.end_date = end_date
        self.time_period_per_sample_hop_in_days = time_period_per_sample_hop_in_days
        self.amount_of_lines_to_sample_per_hop = amount_of_lines_to_sample_per_hop
        self.limit_ratio_lines_to_gtfs_lines = limit_ratio_lines_to_gtfs_lines
        self.limit_routes_per_single_day = limit_routes_per_single_day
        self.limit_stops_per_single_day = limit_stops_per_single_day
        self.amount_of_rides_to_sample_per_hop = amount_of_rides_to_sample_per_hop
        self.lines = lines
        self.id = f"{name}_{start_date}_{end_date}"

    def fetch(self):
        # Check if data already exists
        final_data_path = f"{DATA_DIR}/{self.id}_final_data.csv"
        if os.path.exists(final_data_path):
            return pd.read_csv(final_data_path)

        # Fetch data
        self.data = self.fetch_loop()
        # Save data
        self.cache_table(self.data, "final_data")
        return self.data


    def get_time_periods(self):
        time_periods = []
        # Time example - 2023-08-01T00:00:00+03:00
        start_time_as_datetime = datetime.datetime.strptime(self.start_date, "%Y-%m-%dT%H:%M:%S%z")
        end_time_as_datetime = datetime.datetime.strptime(self.end_date, "%Y-%m-%dT%H:%M:%S%z")

        current_time = start_time_as_datetime
        while current_time < end_time_as_datetime:
            current_time_as_str = current_time.strftime("%Y-%m-%dT%H:%M:%S%z")
            next_time = current_time + timedelta(days=self.time_period_per_sample_hop_in_days)
            next_time_as_str = next_time.strftime("%Y-%m-%dT%H:%M:%S%z")
            time_periods.append((current_time_as_str, next_time_as_str))
            current_time = next_time

        return time_periods
    def fetch_loop(self):
        # One time convert lines to gtfs lines
        self.gtfs_lines = self.convert_lines_to_gtfs_lines()
        self.lines_refs_unq = self.gtfs_lines['line_ref'].unique()

        full_siri_ride_stops = pd.DataFrame()
        for start_time_as_str, end_time_as_str in self.get_time_periods():
            siri_ride_stops_daily = self.fetch_single_time_period(start_time_as_str, end_time_as_str)
            full_siri_ride_stops = pd.concat([full_siri_ride_stops, siri_ride_stops_daily])

        return full_siri_ride_stops

    def fetch_single_time_period(self, start_time_str, end_time_str):
        # Randomly select lines refs to sample
        sampled_line_refs = random.sample(list(self.lines_refs_unq), self.amount_of_lines_to_sample_per_hop)
        print(f"Sampled line refs: {sampled_line_refs}")
        # Get rides for those lines
        siri_rides = self.read_siri_rides(sampled_line_refs, start_time_str, end_time_str)
        # return if siri rides is empty
        if siri_rides.empty:
            return pd.DataFrame()
        siri_ride_ids_unq = siri_rides['id'].unique()
        # Get stops for those rides
        siri_ride_stops = self.read_siri_ride_stops(siri_ride_ids_unq, start_time_str, end_time_str)
        return siri_ride_stops

    def convert_lines_to_gtfs_lines(self):
        cached_data = self.read_cached_table("lines_data")
        if cached_data is not None:
            return cached_data

        lines_data = pd.DataFrame()

        # Iterate each line and append to the lines_data dataframe
        for line in self.lines:
            new_line_by_short_name = pd.DataFrame(
                stride.iterate('/gtfs_routes/list', {"route_short_name": line, "limit": 5}, limit=5))

            lines_data = pd.concat([lines_data, new_line_by_short_name])

        self.cache_table(lines_data, "lines_data")
        return lines_data

    def read_siri_rides(self, line_refs_unq, start_time_str, end_time_str):
        table_name = f"siri_rides_{start_time_str}_{end_time_str}"
        cached_data = self.read_cached_table(table_name)
        if cached_data is not None:
            return cached_data

        # Get GTFS Data
        siri_rides = pd.DataFrame(stride.iterate('/siri_rides/list', {
            "gtfs_ride__start_time_from": start_time_str,
            "scheduled_start_time_from": start_time_str,
            "gtfs_ride__start_time_to": end_time_str,
            "scheduled_start_time_to": end_time_str,
            "gtfs_route__line_refs": ",".join([str(line_ref) for line_ref in line_refs_unq]),
            "limit": self.limit_routes_per_single_day
        }, limit = self.limit_routes_per_single_day))

        self.cache_table(siri_rides, table_name)
        return siri_rides

    def read_siri_ride_stops(self, siri_ride_ids_unq, start_time_str, end_time_str):
        table_name = f"siri_ride_stops_{start_time_str}_{end_time_str}"
        cached_data = self.read_cached_table(table_name)
        if cached_data is not None:
            return cached_data

        # Randomly sample rids ids
        if len(siri_ride_ids_unq) > self.amount_of_rides_to_sample_per_hop:
            print(f"Sampling {self.amount_of_rides_to_sample_per_hop} rides out of {len(siri_ride_ids_unq)}")
            siri_ride_ids_unq = random.sample(list(siri_ride_ids_unq), self.amount_of_rides_to_sample_per_hop)

        # Get GTFS Data
        siri_ride_stops = pd.DataFrame(stride.iterate('/siri_ride_stops/list', {
            "siri_ride_ids": ",".join([str(siri_ride_id) for siri_ride_id in siri_ride_ids_unq]),
            "siri_ride__scheduled_start_time_from": start_time_str,
            "siri_ride__scheduled_start_time_to": end_time_str,
            "limit": self.limit_stops_per_single_day
        }, limit=self.limit_stops_per_single_day))

        self.cache_table(siri_ride_stops, table_name)
        return siri_ride_stops

    def cache_table(self, table, table_name):
        table.to_csv(f"{DATA_DIR}/{self.id}_{table_name}.csv")

    def read_cached_table(self, table_name):
        path = f"{DATA_DIR}/{self.id}_{table_name}.csv"
        # Check if file exists
        if not os.path.exists(path):
            return None
        table = pd.read_csv(path)
        # remove unnamed column
        table = table.loc[:, ~table.columns.str.contains('^Unnamed')]
        return table