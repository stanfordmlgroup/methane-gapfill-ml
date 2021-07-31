import json
import pandas as pd
from pathlib import Path

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

    Writes gapfilled data to data/{SiteID}/gapfilled/{model}_{predictors}.csv
    where the CSV has all the same columns as raw.csv
    plus extra columns for the predicted mean and 95% uncertainty
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
        y_hat_dist_scaled = model_obj.predict_dist(
            gap_df,
            distribution=distribution,
            uncertainty_scale=uncertainty_scale
        )

        y_hat = [dist.mean() for dist in y_hat_dist_scaled]
        # Compute 95% interval
        y_hat_uncertainty_95 = get_pred_interval(y_hat_dist_scaled)
        
        gapfill_columns = {
            f'FCH4_F_{model.upper()}': y_hat,
            f'FCH4_F_UNCERTAINTY_LOWER_{model.upper()}': y_hat_uncertainty_95[:, 0],
            f'FCH4_F_UNCERTAINTY_UPPER_{model.upper()}': y_hat_uncertainty_95[:, 1]
        }
        for column, gapfill_data in gapfill_columns.items():
            gap_df[column] = gapfill_data

        # Merge with original data
        raw_path = site_data_dir / 'raw.csv'
        raw_df = pd.read_csv(raw_path)
        gap_df = raw_df.merge(
            gap_df[list(gapfill_columns.keys()) + ['TIMESTAMP_END']],
            on='TIMESTAMP_END'
        )

        outpath = gap_dir / f"{model}_{predictor_subset}_{distribution}.csv"
        print(f"Writing gapfilled data to {outpath}")
        gap_df.to_csv(outpath, index=False)
