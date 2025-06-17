def build_alert_query(alert_name, init_time, end_time, time_field="Last_occurrence"):
    return {
        "bool": {
            "must": [
                {"match": {"Alert_Name": alert_name}},
                {
                    "range": {
                        time_field: {
                            "gte": init_time,
                            "lte": end_time,
                            "format": "dd-MMMM-yy HH:mm:ss"
                        }
                    }
                }
            ]
        }
    }


def build_iso_time_query(init_time, end_time, time_field="@timestamp"):
    return {
        "range": {
            time_field: {
                "gte": init_time,
                "lte": end_time
            }
        }
    }
