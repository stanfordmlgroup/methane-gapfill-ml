import numpy as np
import pandas as pd


def add_all_predictors(predictor_subset, columns):
    """Add in all predictors in columns to predictor_subset."""
    other_predictors = []
    for predictor in columns:
        if predictor not in predictor_subset and 'FCH4' not in predictor:
           other_predictors.append(predictor)
        elif predictor == 'FCH4':
            continue
        elif 'FCH4' in predictor:
            print(f"Ignoring predictor {predictor}.")

    predictor_subset += other_predictors
    predictor_subset.remove('all')
    
    return predictor_subset


def process_wind_direction_predictor(X):
    X = X.copy()
    X['WD'] = X['WD'] / 180 * np.pi
    notna = ~X['WD'].isna()
    X.loc[notna, 'WD_sin'] = np.sin(X.loc[notna, 'WD'])
    X.loc[notna, 'WD_cos'] = np.cos(X.loc[notna, 'WD'])
    X.drop('WD', axis=1, inplace=True)
    return X


def get_temporal_predictors(timestamp):
    # add sinusoidal functions
    timestamp = pd.to_datetime(timestamp, format='%Y%m%d%H%M').rename('delta')
    doy = timestamp.dt.day
    sin = np.sin(2 * np.pi * (doy - 1) / 365).rename('yearly_sin')
    cos = np.cos(2 * np.pi * (doy - 1) / 365).rename('yearly_cos')

    # add time deltas
    delta = (timestamp - timestamp.iloc[0]) / pd.to_timedelta(1, unit='D')

    return pd.concat([sin, cos, delta], axis=1)
