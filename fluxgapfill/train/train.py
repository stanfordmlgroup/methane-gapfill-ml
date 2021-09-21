import json
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

from fluxgapfill.models import get_model_class
from fluxgapfill.predictors import (
    parse_predictors,
    check_predictors_present,
    add_all_predictors
)


def train(
        sites,
        data_dir,
        models,
        predictors=None,
        predictors_paths=None,
        inner_cv=5,
        n_iter=20,
        log_metrics=["pr2", "nmae"],
        overwrite_existing_models=False,
        **kwargs
):
    """
    Train models using predictors on sites.

    Args:
        sites (list<str>): Comma-separated list of site IDs to train on.
                           Must match the name(s) of the data directories.
        data_dir (<str>): directory of the data folder containing sites folders. 
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
        inner_cv (int): Number of folds for k-fold cross validation in the
                        training set(s) for selecting model hyperparameters.
        n_iter (int): Number of parameter settings that are sampled in the
                      inner cross validation.
        log_metrics (list<str>): Validation metrics to log.
        overwrite_existing_models (bool): Whether to overwrite models if they
                                          already exist.

    Saves trained models to data/{SiteID}/{model}/{predictor}/
    for each {SiteID}, {model}, and {predictor} subset.
    """
    general_args = ['sites', 'models', 'predictors', 'predictors_paths',
                    'overwrite_existing_models']
    args = locals()
    data_dir = Path(data_dir)
    if isinstance(sites, str):
        sites = sites.split(",")
    if isinstance(models, str):
        models = models.split(",")
        for model in models:
            try:
                get_model_class(model)
            except Exception as e:
                raise ValueError(f"Model {model} not supported.")
  
    predictor_subsets = parse_predictors(predictors_paths, predictors)

    for site in sites:
        # Ensure that preprocessing has been run on this site
        site_data_dir = data_dir / site
        train_dir = site_data_dir / "training"
        if not train_dir.exists() or len(list(train_dir.iterdir())) < 1:
            raise ValueError(f"You must run preprocessing on site {site}" +
                             "before training.")

        # Load the training and validation sets
        n_train = len(list(train_dir.glob("train*.csv")))
        train_sets = [
            pd.read_csv(train_dir / f"train{i+1}.csv")
            for i in range(n_train)
        ]
        valid_sets = [
            pd.read_csv(train_dir / f"valid{i+1}.csv")
            for i in range(n_train)
        ]

        save_dir = site_data_dir / "models"
        save_dir.mkdir(exist_ok=True)

        for model in models:
            ModelClass = get_model_class(model)
            for predictor_subset_name, predictor_subset in predictor_subsets.items():
                # Check if predictor exists in the data
                for data_set in train_sets + valid_sets:
                    check_predictors_present(data_set, predictor_subset)

                model_dir = save_dir / model / predictor_subset_name
                model_dir.mkdir(exist_ok=True, parents=True)
                args_path = model_dir / "args.json"
                if (
                        len(list(model_dir.glob("*.pkl"))) > 0
                        and not overwrite_existing_models
                ):
                    with args_path.open() as f: ## TODO: double-check model matching detection
                        prev_args = json.load(f)
                    # If args don't match, raise a ValueError and ask that
                    # the user specifies overwrite_existing_models=True.
                    for key, value in args.items():
                        if key in general_args:
                            continue
                        prev_value = prev_args[key]
                        if prev_value != value:
                            raise ValueError(
                                f"You supplied {key}={value} but the saved model" +
                                f" used {key}={prev_value}. Please specify "
                                "overwrite_existing_models=True to overwrite."
                            )
                else:
                    # Either model doesn't exist or
                    # overwrite_existing_models=True
                    # Write args to file
                    with args_path.open('w') as f:
                        json.dump(args, f)

                predictor_subset_print = predictor_subset
                if 'all' in predictor_subset_print:
                    predictor_subset_print = add_all_predictors(
                        predictor_subset_print, train_sets[0].columns
                    )

                print(f"Model training...\n" +
                      f" - site: {site}\n" +
                      f" - model: {model}\n" +
                      f" - predictors: {','.join(predictor_subset_print)}")

                scores = defaultdict(list)
                for i, (train_set, valid_set) in enumerate(zip(train_sets,
                                                                    valid_sets)):
                    print(f' - Training on {i}/{len(train_sets)}...')
                    model_path = model_dir / f"model{i+1}.pkl"
                    if model_path.exists() and not overwrite_existing_models:
                        print(f"  - Loading existing model from {model_path}.")
                        with model_path.open("rb") as f:
                            model_obj = pkl.load(f)
                    else:
                        model_obj = ModelClass(predictor_subset=predictor_subset,
                                               cv=inner_cv, n_iter=n_iter)
                        model_obj.fit(train_set, train_set['FCH4'])

                    for metric in log_metrics:
                        score = model_obj.evaluate(valid_set, valid_set['FCH4'], metric)
                        scores[metric].append(score)

                    model_obj.save(model_path)

                scores_df = pd.DataFrame(scores)
                scores_path = model_dir / "training_results.csv"
                scores_df.to_csv(scores_path, index=False)

            print(f" - Done model training.\n")
