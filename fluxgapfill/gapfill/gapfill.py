import os
import json
import numpy as np
import pandas as pd
import datetime as dt
from pathlib import Path
from collections import defaultdict
from dateutil.relativedelta import relativedelta

from fluxgapfill.models import EnsembleModel
from fluxgapfill.test import parse_model_dirs
from fluxgapfill.predictors import parse_predictors
from fluxgapfill.metrics import get_pred_interval
from fluxgapfill.models import get_model_class

def gapfill(
        data_dir,
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
        # model_dirs (list<str>): Comma-separated list of paths to model
        #                         directories with checkpoints. These must all
        #                         be from the same site. sites, models,
        #                         predictors, and predictors_paths are ignored
        #                         if this parameter is supplied to the function.
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
                                       ranges. See
                                           gapfill/budget_date_ranges.json
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
    # parse inputs
    if isinstance(sites, str):
        sites = sites.split(",")
    if isinstance(models, str):
        models = models.split(",")
        for model in models:
            try:
                get_model_class(model)
            except Exception as e:
                raise ValueError(f"Model {model} not supported.")
    
    
    if (predictors is not None) and (predictors_paths is not None): 
        predictor_subsets = parse_predictors(predictors_paths, predictors)   # list?
    else:
        predictor_subsets = None

    if budget_date_ranges_path is not None:
        with Path(budget_date_ranges_path).open() as f:
            budget_date_ranges = json.load(f)
    else:
        budget_date_ranges = {}
    
    # 
    data_dir = Path(data_dir)
    if sites is None:
        sites = os.listdir(data_dir)
        if "results" in sites:
            sites.remove("results")
    
    for site in sites:
        site_dir = data_dir/site
        site_gap_dir = site_dir / "gapfilled"
        site_gap_dir.mkdir(exist_ok=True)
        
        if models is None:
            models = os.listdir(site_dir/'models')
        
        for model in models:
            if predictor_subsets is None:
                predictor_subsets = os.listdir(site_dir/'models'/model)
                
            for predictor_subset in predictor_subsets:
                print(f'Gapfilling: {site}, {model}, {predictor_subset}, {distribution}')
                gap_df = gapfill_site_model_predictor(site, 
                                                      model, 
                                                      predictor_subset, 
                                                      distribution, 
                                                      site_dir)
                out_fn = site_gap_dir / f"{model}_{predictor_subset}_{distribution}.csv"
                print(f" - Writing gapfilled data to {out_fn}\n")
                gap_df.to_csv(out_fn, index=False)
               
                ## compute annual budget
                print(f'Computing Annual Budget: {site}, {model}, {predictor_subset}, {distribution}')
                # For budget estimates, convert from nmol m-2 s-1 to g C m-2 halfhour-1
                ch4_conversion = lambda fch4: fch4*60*30*12.0107*10**-9
                for col in [col for col in gap_df.columns if 'FCH4' in col]:
                    gap_df[col] = gap_df[col].apply(ch4_conversion)
                
                site_budget_date_ranges = get_site_budget_ranges(site, budget_date_ranges, gap_df)
                budget_df = compute_annual_budget(site, gap_df, 
                                                  site_budget_date_ranges)

                # Write budget data to file
                out_fn = (
                    site_gap_dir / f"{model}_{predictor_subset}_{distribution}_budget.csv"
                )
                print(f" - Writing budget data to {out_fn}\n")
                budget_df.to_csv(out_fn, index=False)


def gapfill_site_model_predictor(site, model, predictor, distribution, site_dir):
        
    model_dir = site_dir/"models"/model/predictor
    Model = EnsembleModel(model_dir)
    
    gap_df = pd.read_csv(site_dir / 'gap.csv')
    scale_fn = model_dir / "scale.json"
    if not scale_fn.exists():
        raise ValueError(
            "Must run <python main.py test> with --distribution " +
            f"{distribution} to compute an uncertainty scale before " +
            "gapfilling."
        )
    with scale_fn.open() as f:
        uncertainty_scale = json.load(f)

    y_hat_individual = Model.predict_individual(gap_df)
    y_hat_mean = np.mean(y_hat_individual, axis=0)
    y_hat_spread = (
        (y_hat_individual - y_hat_mean) * uncertainty_scale + y_hat_mean
    )
    y_hat_dist = Model.predict_dist(
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
    raw_fn = site_dir / 'raw.csv'
    raw_df = pd.read_csv(raw_fn)

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

    # Set uncertainties around observed values to 0
    gap_df.loc[observed_rows, 'FCH4_F_UNCERTAINTY'] = 0
    return gap_df


def get_site_budget_ranges(site, budget_date_ranges, gap_df):
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
    return site_budget_date_ranges
       

def compute_annual_budget(site, gap_df, site_budget_date_ranges):

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
        
        fch4_columns = [col for col in gap_df.columns if 'FCH4_F' in col]
        fch4_columns.remove('FCH4_F_UNCERTAINTY')
        annual_budgets = annual_gap_df[fch4_columns].sum()

        budget_mean = annual_budgets.mean()
        budget_uncertainty = annual_budgets.std() * 1.96

        date_range_str = f"{range_start.date()} and {range_end.date()}"
        budget_str = f"{budget_mean:.2f} +/- {budget_uncertainty:.2f}"
        print(f" - Budget between {date_range_str} is {budget_str}")

        budget_dict['range_start'].append(range_start)
        budget_dict['range_end'].append(range_end)
        budget_dict['budget_mean'].append(budget_mean)
        budget_dict['budget_uncertainty'].append(budget_uncertainty)

    budget_df = pd.DataFrame(budget_dict)

    # Combine budgets and uncertainties per year with an average
    mean_budget_dict = {
        'range_start': site_budget_date_ranges[0][0],
        'range_end': site_budget_date_ranges[-1][1],
        'budget_mean': budget_df['budget_mean'].mean(),
        'budget_uncertainty': budget_df['budget_uncertainty'].mean()
    }
    budget_df = budget_df.append(mean_budget_dict, ignore_index=True)
    return budget_df

