import json
import numpy as np
import pandas as pd
import datetime as dt
from pathlib import Path
from collections import defaultdict
from dateutil.relativedelta import relativedelta

from models import EnsembleModel
from test import parse_model_dirs
from metrics import get_pred_interval


def gapfill(
        model_dirs=None,
        sites=None,
        models=None,
        predictors=None,
        predictors_paths=None,
        distribution="laplace",
        budget_date_ranges_path=None,
        **kwargs
):
    """
    Gapfill data with trained models.

    Args:
        model_dirs (list<str>): Comma-separated list of paths to model
                                directories with checkpoints. These must all
                                be from the same site. sites, models,
                                predictors, and predictors_paths are ignored
                                if this parameter is supplied to the function.
        sites (list<str>): Comma-separated list of site IDs to train on.
                           Must match the name(s) of the data directories.
        models (list<str>): Comma-separated list of model names to train.
                            Options: ['rf', 'ann', 'lasso', 'xgb']
        predictors (list<str>): Comma-separated list of predictors. Ignored if
                                predictors_path is provided.
                                Certain keyword predictors are used to denote
                                specific sets of predictors:
                                ['temporal', 'all']
        predictors_paths (list<str>): Comma-separated list of paths to files
                                      containing predictor names. See
                                      predictors/metereological.txt for an
                                      example.
        distribution (str): Which distribution to use for prediction.
                            Options: ['laplace', 'normal']
        budget_date_ranges_path (str): Dictionary mapping site names to date
                                       ranges. See gapfill/budget_date_ranges.json
                                       for an example.

    Writes gapfilled data to
        data/{SiteID}/gapfilled/{model}_{predictors}_{distribution}.csv
    where the CSV has all the same columns as raw.csv, excluding any existing
    gapfilled columns, plus columns for the
        predicted mean (FCH4_F)
        95% uncertainty (FCH4_uncertainty)
        spread individual predictions (FCH4{1-N})
    and writes budget data to 
        data/{SiteID}/gapfilled/{model}_{predictors}_{distribution}_budget.csv
    """
    data_dir = Path("data/")
    model_dirs = parse_model_dirs(
        model_dirs=model_dirs,
        sites=sites,
        models=models,
        predictors=predictors,
        predictors_paths=predictors_paths,
        data_dir=data_dir
    )
    if budget_date_ranges_path is not None:
        with Path(budget_date_ranges_path).open() as f:
            budget_date_ranges = json.load(f)
    else:
        budget_date_ranges = {}

    for model_dir in model_dirs:
        model_dir = Path(model_dir)
        site_data_dir = model_dir.parent.parent.parent

        predictor_subset = model_dir.name
        model = model_dir.parent.name
        site = site_data_dir.name

        gap_dir = site_data_dir / "gapfilled"
        gap_dir.mkdir(exist_ok=True)

        gap_path = site_data_dir / 'gap.csv'
        gap_df = pd.read_csv(gap_path)
        
        model_obj = EnsembleModel(model_dir)

        scale_path = model_dir / "scale.json"
        if not scale_path.exists():
            raise ValueError(
                "Must run <python main.py test> with --distribution " +
                f"{distribution} to compute an uncertainty scale before " +
                "gapfilling."
            )
        with scale_path.open() as f:
            uncertainty_scale = json.load(f)

        y_hat_individual = model_obj.predict_individual(gap_df)
        y_hat_mean = np.mean(y_hat_individual, axis=0)
        y_hat_spread = (
            (y_hat_individual - y_hat_mean) * uncertainty_scale + y_hat_mean
        )
        y_hat_dist = model_obj.predict_dist(
            gap_df,
            distribution=distribution,
            uncertainty_scale=uncertainty_scale
        )
        y_hat_uncertainty_95 = get_pred_interval(y_hat_dist)

        num_models = len(y_hat_individual)
        gapfill_columns = {
            'FCH4_F': y_hat_mean,
            'FCH4_F_UNCERTAINTY': y_hat_uncertainty_95[:, 1] - y_hat_mean,
            **{
                f"FCH4_F{i+1}": y_hat_spread[i]
                for i in range(num_models)
            }
        }
        for column, gapfill_data in gapfill_columns.items():
            gap_df[column] = gapfill_data

        # Merge with original data
        raw_path = site_data_dir / 'raw.csv'
        raw_df = pd.read_csv(raw_path)

        # Drop existing columns from the raw df
        existing_fch4_columns = [
            column for column in raw_df.columns
            if 'FCH4_' in column
        ]
        raw_df.drop(columns=existing_fch4_columns, inplace=True)
        gap_df = raw_df.merge(
            gap_df[list(gapfill_columns.keys()) + ['TIMESTAMP_END']],
            on='TIMESTAMP_END',
            how='left'
        )

        # Add in observed values to FCH4_F, FCH4_F{1-N}
        fch4_columns = [f'FCH4_F{i+1}' for i in range(num_models)]
        observed_rows = gap_df['FCH4_F'].isna()
        for fch4_column in ['FCH4_F'] + fch4_columns:
            gap_df.loc[observed_rows, fch4_column] = (
                gap_df.loc[observed_rows, 'FCH4']
            )
        # TODO: Should we keep uncertainty around observed values?
        gap_df.loc[observed_rows, f'FCH4_F_UNCERTAINTY'] = 0

        outpath = gap_dir / f"{model}_{predictor_subset}_{distribution}.csv"
        print(f"Writing gapfilled data to {outpath}")
        gap_df.to_csv(outpath, index=False)

        # For budget estimates, convert from nmol m-2 s-1 to g C m-2 halfhour-1
        for fch4_column in fch4_columns + ['FCH4', 'FCH4_F_UNCERTAINTY']:
            gap_df[fch4_column] = gap_df[fch4_column].apply(
                lambda fch4: fch4*60*30*12.0107*10**-9
            )

        # Get date ranges to compute annual budget
        if site in budget_date_ranges:
            site_budget_date_ranges = [
                [
                    dt.datetime(
                        int(range_start.split('-')[1]),
                        int(range_start.split('-')[0]),
                        1
                    ),
                    dt.datetime(
                        int(range_end.split('-')[1]),
                        int(range_end.split('-')[0]),
                        1
                    ) + relativedelta(months=1)
                ]
                for [range_start, range_end] in budget_date_ranges[site]
            ]
        else:
            annual_sum_year_start = gap_df['Year'].min()
            annual_sum_year_end = gap_df['Year'].max()
            site_budget_date_ranges = [
                [
                    dt.datetime(year, 1, 1),
                    dt.datetime(year+1, 1, 1)
                ]
                for year in range(
                    annual_sum_year_start,
                    annual_sum_year_end + 1
                )
            ]

        # Compute annual budgets between date ranges
        gap_df["TIMESTAMP_END"] = pd.to_datetime(
            gap_df["TIMESTAMP_END"],
            format='%Y%m%d%H%M'
        )
        budget_dict = defaultdict(list)
        for [range_start, range_end] in site_budget_date_ranges:
            annual_gap_df = gap_df[
                (gap_df['TIMESTAMP_END'] >= range_start) &
                (gap_df['TIMESTAMP_END'] < range_end)
            ]
            annual_budgets = annual_gap_df[fch4_columns].sum()

            budget_mean = annual_budgets.mean()
            budget_uncertainty = annual_budgets.std() * 1.96
            print(
                f"Budget between {range_start.date()} and {range_end.date()}" +
                f" is {budget_mean:.2f} +/- {budget_uncertainty:.2f}"
            )

            budget_dict['range_start'].append(range_start)
            budget_dict['range_end'].append(range_end)
            budget_dict['budget_mean'].append(budget_mean)
            budget_dict['budget_uncertainty'].append(budget_uncertainty)

        # TODO: How did we combine uncertainties per year? average?

        # Write budget data to file
        budget_df = pd.DataFrame(budget_dict)
        outpath = (
            gap_dir / f"{model}_{predictor_subset}_{distribution}_budget.csv"
        )
        print(f"Writing budget data to {outpath}")
        budget_df.to_csv(outpath, index=False)
