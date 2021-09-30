import pandas as pd


def load_raw_data(site_data_path, na_values=-9999):
    print(' - Loading site data')
    if not site_data_path.exists():
        raise ValueError(f"Expected data at {site_data_path} but " +
                         "the data was not found")
    site_data = pd.read_csv(site_data_path, na_values=na_values)
    
    expected_columns = [
        "TIMESTAMP_END",
        "FCH4"
    ]
    for expected_column in expected_columns:
        if expected_column not in site_data.columns:
            raise ValueError(f"CSV is missing column {expected_column}.")

    try:
        datetimes = pd.to_datetime(site_data["TIMESTAMP_END"], format='%Y%m%d%H%M')
    except:
        raise ValueError("TIMESTAMP_END needs to be formatted as " +
                         "`YYYYMMDDHHmm`. Also check if any 00 exists in %d")
    
    if "year" not in site_data.columns: # required by budget estimation
        site_data["year"] = [dt.year for dt in datetimes]
        
    return site_data


def load_test_data():
    '''
    load testing data from Github.
    '''

    print(' - Loading test data')
    
    fn = 'https://raw.githubusercontent.com/ylzhouchris/ylzhou/a5dfd24f92fb2b3cf1abdb886567d82b56ed9b5b/test_data/raw.csv'
    try: 
        test_data = pd.read_csv(fn)
    except:
        raise ValueError("Error retrieving testing data from Github. Please check Internet connection.")
    return test_data
