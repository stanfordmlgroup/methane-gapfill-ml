import numpy as np
import pandas as pd
import pickle as pkl
from pathlib import Path
from collections import defaultdict

from models import get_model_class


def train(
        sites,
        models,
        predictors=None,
        predictors_paths=None,
        log_metrics=["pr2", "nmae"],
        inner_cv=5,
        n_iter=20,
        overwrite_existing_models=False
):
    """
    Train models using predictors on sitees.

    Args:
        sites (list<str>): Comma-separated list of site IDs to train on.
                           Must match the name(s) of the data directories.
        models (list<str>): Comma-separated list of model names to train.
                            Options: ['rf', 'ann', 'lasso', 'xgb']
        predictors (list<str>): Comma-separated list of predictors. Ignored if
                                predictors_path is provided.
        predictors_paths (list<str>): Comma-separated list path file(s) with
                                      predictor names. See train/predictors.txt
                                      for an example.
        log_metrics (list<str>): Validation metrics to log.
        inner_cv (int): Number of folds for k-fold cross validation in the
                        training set(s) for selecting model hyperparameters.
        n_iter (int): Number of parameter settings that are sampled in the
                      inner cross validation.
        overwrite_existing_models (bool): Whether to overwrite models if they
                                          already exist.

    Saves trained models to data/{SiteID}/{model}/{predictor}/
    for each {SiteID}, {model}, and {predictor} subset.
    """
    data_dir = Path("data/")
    if isinstance(sites, str):
        sites = sites.split(",")
    if isinstance(models, str):
        models = models.split(",")
    for model in models:
        try:
            get_model_class(model)
        except Exception as e:
            raise ValueError(f"Model {model} not supported.")

    if predictors_paths is not None:
        if isinstance(predictors_paths, str):
            predictors_paths = predictors_paths.split(",")
        predictor_subsets = {}
        for predictors_path in predictors_paths:
            predictors_path = Path(predictors_path)
            with predictors_path.open() as f:
                predictor_subset = f.read().splitlines()
            predictor_subsets[predictors_path.stem] = predictor_subset
    elif predictors is not None:
        if isinstance(predictors, str):
            predictors = predictors.split(",")
        predictor_subsets = {"predictors": predictors}
    else:
        raise ValueError("Must provide predictors or predictors_paths.")

    for site in sites:
        # Ensure that preprocessing has been run on this site
        site_data_dir = data_dir / site
        train_dir = site_data_dir / "training"
        if not train_dir.exists() or len(list(train_dir.iterdir())) < 1:
            raise ValueError(f"You must run preprocessing on site {site}" +
                             "before training.")

        # Load the training and validation sets
        train_sets = [
            pd.read_csv(train_path)
            for train_path in sorted(train_dir.glob("train*.csv"))
        ]
        valid_sets = [
            pd.read_csv(valid_path)
            for valid_path in sorted(train_dir.glob("valid*.csv"))
        ]

        site_scores = {}
        for predictor_subset_name, predictor_subset in predictor_subsets.items():
            # Check if predictor exists in the data
            for data_set in train_sets + valid_sets:
                for predictor in predictor_subset:
                    if predictor not in data_set.columns:
                        raise ValueError(f"Predictor {predictor} not found " +
                                         "in the data.")

            print(f"Training models for site={site} with " +
                  f"predictors={','.join(predictor_subset)}...")
            model_scores = defaultdict(lambda: defaultdict(list))
            for i, (train_set, valid_set) in enumerate(zip(train_sets,
                                                           valid_sets)):

                # TODO: Process special keyword predictors

                X_train = train_set[predictor_subset].copy()
                y_train = train_set["FCH4"]
                X_valid = valid_set[predictor_subset].copy()
                y_valid = valid_set["FCH4"]

                for model in models:
                    model_dir = site_data_dir / model / predictor_subset_name
                    model_dir.mkdir(exist_ok=True, parents=True)
                    model_path = model_dir / f"model{i+1}.pkl"
                    if model_path.exists() and not overwrite_existing_models:
                        print(f"Loading existing model from {model_path}")
                        with model_path.open("rb") as f:
                            model_obj = pkl.load(f)
                    else:
                        ModelClass = get_model_class(model)
                        model_obj = ModelClass(cv=inner_cv, n_iter=n_iter)
                        model_obj.fit(X_train, y_train)

                    for metric in log_metrics:
                        score = model_obj.evaluate(X_valid, y_valid, metric)
                        model_scores[model][metric].append(score)

                    model_obj.save(model_path)
            site_scores[predictor_subset_name] = model_scores

        # Aggregate scores in a pandas dataframe
        site_scores_mean = defaultdict(lambda: defaultdict(dict))
        for predictor_subset_name, model_scores in site_scores.items():
            for model, metric_scores in model_scores.items():
                for metric, scores in metric_scores.items():
                    mean_score = np.mean(scores)
                    site_scores_mean[predictor_subset_name][model][metric] = mean_score

        site_scores_df = pd.concat({
            predictor_subset_name: pd.DataFrame(model_scores)
            for predictor_subset_name, model_scores in site_scores_mean.items()
        }, axis=0)
        site_scores_df.index = site_scores_df.index.rename(
            ["predictors", "metric"]
        )

        # Write the scores to file
        site_scores_dir = site_data_dir / "results"
        site_scores_dir.mkdir(exist_ok=True)
        site_scores_path = site_scores_dir / "valid.csv"
        print(f"Done training models for site={site}.")
        print(f"Writing validation results to {site_scores_path}")
        site_scores_df.to_csv(site_scores_path)
