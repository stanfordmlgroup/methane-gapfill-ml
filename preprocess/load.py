import pandas as pd


def load_raw_data(site_data_path):
    if not site_data_path.exists():
        raise ValueError(f"Expected data at {site_data_path} but " +
                         "the data was not found")
    site_data = pd.read_csv(site_data_path)
    
    expected_columns = [
        "TIMESTAMP_END",
        "FCH4"
    ]
    for expected_column in expected_columns:
        if expected_column not in site_data.columns:
            raise ValueError(f"CSV is missing column {expected_column}.")

    try:
        pd.to_datetime(site_data["TIMESTAMP_END"], format='%Y%m%d%H%M')
    except:
        raise ValueError("TIMESTAMP_END needs to be formatted as " +
                         "`YYYYMMDDHHmm`")

    return site_data
