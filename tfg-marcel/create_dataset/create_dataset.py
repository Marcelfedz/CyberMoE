import os
#import fireducks.pandas as pd
import pandas as pd
import pickle
import pytz

from elasticsearch import Elasticsearch
from dotenv import load_dotenv
from datetime import datetime, timedelta
from q_db import build_alert_query, build_iso_time_query


class ttp_dataset:
    def __init__(self):
        self.es = self.connect_elasticsearch()

    def connect_elasticsearch(self):
        load_dotenv()
        es_host = os.getenv("ELASTICSEARCH_HOST")
        es_user = os.getenv("ELASTICSEARCH_USER")
        es_password = os.getenv("ELASTICSEARCH_PASSWORD")
        es = Elasticsearch(es_host,basic_auth=(es_user, es_password),)
        if es.ping():
            return es
        else:
            print("Could not connect to Elasticsearch.")
            return None
    
    def get_alerts(self, index_name, alert_name, init_time, end_time):
        query = build_alert_query(alert_name, init_time, end_time)
        response = self.es.search(
            index=index_name,
            query=query,
            size=10000,
            _source=["Alert_Name", "User", "Last_occurrence", "RecordNumber"] 
        )
        hits = response.get("hits", {}).get("hits", [])
        results = []
        for hit in hits:
            src = hit.get("_source", {})
            if all(k in src for k in ["Alert_Name", "User", "Last_occurrence"]):
                results.append({
                    "Alert_Name": src["Alert_Name"],
                    "User": src["User"],
                    "Last_occurrence": src["Last_occurrence"],
                    "RecordNumber": src["RecordNumber"]  
                })

        df = pd.DataFrame(results)
        if not df.empty:
            df["Last_occurrence"] = pd.to_datetime(df["Last_occurrence"], errors="coerce")
            df = df.sort_values("Last_occurrence")
        
        return df

   
    def flatten_dict(self, d, parent_key='', sep='.'):
        """Recursively flattens a nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self.flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    
    def filter_columns(self,df):
        # List of columns to keep
        required_columns = [ 
            '@timestamp', 
            'DescriptionTitle', 
            'Channel', 
            'Task', 
            'ProcessID', 
            'ThreadID', 					
            'EventID',
            'http.request.body.bytes',
            'EventRecordID',

            'subject.account_name',
            'creator_subject.account_name',
            'service_information.service_name',
            'account_information.account_name',
            'new_logon.account_name',
            'account_whose_credentials_were_used.account_name',
            'target_subject.account_name'

        ]
    
        # Only keep columns that exist to avoid KeyError
        existing_columns = [col for col in required_columns if col in df.columns]
        df_filtered = df[existing_columns]

        if missing := list(set(required_columns) - set(existing_columns)):
            print(f"⚠️ Warning: Missing columns not found in logs: {missing}")
        
        return df_filtered
    
    # .
    def save_to_parquet(self, df, save_path):
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)  
        df.to_parquet(save_path, index=False)
    
    
    def fetch_logs_by_time(self, index_name, init_time, end_time, save_path):
        query = build_iso_time_query(init_time, end_time)
        size = 10000
        all_hits = []
        search_after = None
        total = 0

        while True:
            body = {
                "size": size,
                "query": query,
                "sort": [
                    {"@timestamp": "asc"},  
                ]
            }

            if search_after:
                body["search_after"] = search_after

            response = self.es.search(index=index_name, body=body)
            #print(response)
            hits = response['hits']['hits']
            if not hits:
                break

            all_hits.extend([hit["_source"] for hit in hits])
            total += len(hits)

            # Update `search_after` with sort values from the last hit
            search_after = hits[-1]["sort"]

            print(f"Fetched {total} logs so far...")

        if not all_hits:
            print("No logs found in the given time range.")
            return

        # df = pd.DataFrame(all_hits)
        flattened_hits = [self.flatten_dict(hit) for hit in all_hits]
        df = pd.DataFrame(flattened_hits)
        df_filtered = self.filter_columns(df)
        self.save_to_parquet(df=df_filtered, save_path=save_path)
        print(f"Saved {len(df_filtered)} records to {save_path}")
        return df_filtered
 
    # If we normalize the time of the alert, we can compare the timestamps with the upc-logs
    def normalize_alert_times(self, df_alerts: pd.DataFrame, alert_time_col: str = "Last_occurrence") -> pd.DataFrame:
        """
        Converts the alert time column to UTC-aware datetime, matching log timestamps.
        Assumes original format like '04-November-24 18:04:11'.
        """
        df_alerts = df_alerts.copy()

        # Define the datetime format in Last_occurance
        datetime_format = "%d-%B-%y %H:%M:%S"

        # Parse with strptime and localize to UTC
        df_alerts[alert_time_col] = pd.to_datetime(
            df_alerts[alert_time_col], 
            format=datetime_format, 
            errors='coerce'
        ).dt.tz_localize("UTC")  # Make it tz-aware in UTC

        return df_alerts

    def label_logs_with_alerts(
        self,
        df_logs: pd.DataFrame,
        df_alerts: pd.DataFrame,
        log_timestamp_col: str = "@timestamp",
        alert_user_col: str = "User",
        alert_name_col: str = "Alert_Name",
        alert_time_col: str = "Last_occurrence",
        matching_fields: list = None,
        time_window_minutes: int = 10,
        label_col: str = "alert_label"
        ) -> pd.DataFrame:
        
        """
        Labels logs based on alert user matches within a time window.
        Appends new labels to existing ones.
        """
        df_alerts.columns = df_alerts.columns.str.strip()
        df_logs.columns = df_logs.columns.str.strip()

        if alert_time_col not in df_alerts.columns:
            raise ValueError(f"'{alert_time_col}' not found in df_alerts. Available columns: {df_alerts.columns.tolist()}")

        if matching_fields is None:
            matching_fields = [
                "creator_subject.account_name",
                "service_information.service_name",
                "subject.account_name",
                "account_information.account_name",
                "account_whose_credentials_were_used.account_name",
                "new_logon.account_name"
            ]

        df_logs = df_logs.copy()
        df_logs[log_timestamp_col] = pd.to_datetime(df_logs[log_timestamp_col], errors='coerce')
        df_alerts = df_alerts.copy()
        df_alerts[alert_time_col] = pd.to_datetime(df_alerts[alert_time_col], errors='coerce')

        if label_col not in df_logs.columns:
            df_logs[label_col] = None

        for _, alert in df_alerts.iterrows():
            user = alert.get(alert_user_col)
            alert_name = alert.get(alert_name_col)
            alert_time = alert.get(alert_time_col)

            if pd.isnull(alert_time) or pd.isnull(user):
                continue

            end_time = alert_time + timedelta(minutes=time_window_minutes)
            time_mask = (df_logs[log_timestamp_col] >= alert_time) & (df_logs[log_timestamp_col] <= end_time)
            subset = df_logs[time_mask]

            if subset.empty:
                continue

            match_mask = subset[matching_fields].eq(user).any(axis=1)
            matched_indices = subset[match_mask].index

            for idx in matched_indices:
                current_label = df_logs.at[idx, label_col]
                if pd.isna(current_label) or current_label == "":
                    df_logs.at[idx, label_col] = alert_name
                elif alert_name not in current_label.split(","):
                    df_logs.at[idx, label_col] = f"{current_label},{alert_name}"

        return df_logs