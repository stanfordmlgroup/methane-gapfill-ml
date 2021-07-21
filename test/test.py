import json
import pickle
import pandas as pd
from pathlib import Path
from collections import defaultdict

from models import EnsembleModel
from metrics import metric_dict, uncertainty_metric_dict
from predictors import parse_predictors


def test(
        model_dirs=None,
        sites=None,
        models=None,
        predictors=None,
        predictors_paths=None,
        split='test',
        distribution="laplace",
        eval_metrics=list(metric_dict.keys()),
        uncertainty_eval_metrics=list(uncertainty_metric_dict.keys()),
        overwrite_results=False
):
    """
    Evaluate models.

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
        split (str): Which split to test on.
                     Options: ['train', 'valid', 'test']
        distribution (str): Which distribution to use for prediction.
                            Options: ['laplace', 'normal']
        eval_metrics (list<str>): Metrics to use to evaluate the model(s) on
                                  the split.
        uncertainty_eval_metrics (list<str>): Metrics to use to evaluate the
                                              uncertainty estimates of the
                                              model(s) on the test set.
        overwrite_results (bool): Whether to overwrite existing results.

    Writes performance metrics to data/{site}/results/{split}
    """
    args = locals()
    data_dir = Path("data/")
    if isinstance(model_dirs, str):
        model_dirs = model_dirs.split(",")
    if isinstance(sites, str):
        sites = sites.split(",")
    if isinstance(models, str):
        models = models.split(",")

    predictor_subsets = parse_predictors(predictors_paths, predictors)

    if model_dirs is None:
        model_dirs = []
        for site in sites:
            save_dir = data_dir / site / "models"
            for model in models:
                for predictor_subset, predictors in predictor_subsets.items():
                    model_dir = save_dir / model / predictor_subset
                    if len(list(model_dir.glob("*.pkl"))) == 0:
                        raise ValueError(
                            "No models trained for " +
                            f"site={site}, " +
                            f"model={model}, " +
                            f"predictor_subset={predictor_subset}, " +
                            f"predictors={predictors}"
                        )
                    model_dirs.append(model_dir)

    splits = ['train', 'valid', 'test']
    if split not in splits:
        raise ValueError(f"Got split={split} but must be one of " +
                         f"{','.join(splits)}")

    if split != 'test' and scale_uncertainty:
        raise ValueError('Can only scale uncertainties on the test set.')

    eval_scores_dir = Path("results")
    eval_scores_dir.mkdir(exist_ok=True)

    eval_scores_path = eval_scores_dir / f"{split}.csv"
    if eval_scores_path.exists() and not overwrite_results:
        raise ValueError(f"Results path at {eval_scores_path} already exists.")

    eval_scores = []
    for model_dir in model_dirs:
        model_dir = Path(model_dir)
        site_data_dir = model_dir.parent.parent.parent

        predictor_subset = model_dir.name
        model = model_dir.parent.name

        if split == 'test':
            eval_df = pd.read_csv(site_data_dir / 'test.csv')
            model_obj = EnsembleModel(model_dir)
            y = eval_df['FCH4']
            y_hat_dist = model_obj.predict_dist(
                eval_df,
                distribution=distribution
            )
            y_hat = [dist.mean() for dist in y_hat_dist]
            y_hat_scale = [dist.scale for dist in y_hat_dist]
            uncertainty_scale = model_obj.uncertainty_scale(
                y, y_hat_dist, distribution=distribution
            )
            y_hat_dist_scaled = model_obj.predict_dist(
                eval_df,
                distribution=distribution,
                uncertainty_scale=uncertainty_scale
            )
            y_hat_scale_scaled = [dist.scale for dist in y_hat_dist_scaled]
            pred_df = pd.DataFrame({
                "groundtruth": y,
                "prediction": y_hat,
                "uncertainty_scale": y_hat_scale,
                "scaled_uncertainty_scale": y_hat_scale_scaled
            })

            pred_df.to_csv(model_dir / f"{split}_predictions.csv", index=False)
            scores = {
                eval_metric: [metric_dict[eval_metric](y, y_hat)]
                for eval_metric in eval_metrics
            }
            uncertainty_scores = {
                unc_eval_metric: [
                    uncertainty_metric_dict[unc_eval_metric](y, y_hat_dist)
                ]
                for unc_eval_metric in uncertainty_eval_metrics
            }
            scaled_uncertainty_scores = {
                f"{unc_eval_metric}_scaled": [
                    uncertainty_metric_dict[unc_eval_metric](
                        y, y_hat_dist_scaled
                    )
                ]
                for unc_eval_metric in uncertainty_eval_metrics
            }
            scores = {
                **scores,
                **uncertainty_scores,
                **scaled_uncertainty_scores
            }

            scores['model'] = model
            scores['predictors_subset'] = predictor_subset
            predictors = [
                predictor
                for predictor in model_obj.predictors
                if predictor != "intercept"
            ]
            scores['predictors'] = ";".join(predictors)
            eval_scores.append(pd.DataFrame(scores))

        else:
            # Run each model on the corresponding data split
            site_training_data_dir = site_data_dir / 'training'
            n_train = len(list(site_training_data_dir.glob("train*.csv")))

            scores = defaultdict(list)
            for i in range(n_train):
                eval_path = site_training_data_dir / f'{split}{i+1}.csv'
                eval_df = pd.read_csv(eval_path)

                model_path = model_dir / f"model{i+1}.pkl"
                if not model_path.exists():
                    raise ValueError(
                        f"Model path {model_path} does not exist."
                    )
                with model_path.open('rb') as f:
                    model_obj = pickle.load(f)

                y_hat = model_obj.predict(eval_df)
                y = eval_df['FCH4']
                pred_df = pd.DataFrame({"groundtruth": y, "prediction": y_hat})
                pred_df.to_csv(
                    model_dir / f"{split}{i+1}_predictions.csv",
                    index=False
                )
                scores = {
                    eval_metric: [metric_dict[eval_metric](y, y_hat)]
                    for eval_metric in eval_metrics
                }

            scores_df = pd.DataFrame(scores)
            mean_scores = scores_df.mean().to_frame().T
            mean_scores['model'] = model
            mean_scores['predictor_subset'] = predictor_subset
            predictors = [
                predictor
                for predictor in model_obj.predictors
                if predictor != "intercept"
            ]
            mean_scores['predictors'] = ";".join(predictors)
            eval_scores.append(mean_scores)

        eval_scores_df = pd.concat(eval_scores)
        print(f"Adding ongoing {split} metrics for {model_dir} " +
              f"to {eval_scores_path}")
        eval_scores_df.to_csv(eval_scores_path, index=False)

    eval_scores_df = pd.concat(eval_scores)
    print(f"Writing final {split} metrics to {eval_scores_path}")
    eval_scores_df.to_csv(eval_scores_path, index=False)

    # Save args to eval_scores_dir
    eval_args_path = eval_scores_dir / "args.json"
    with eval_args_path.open('w') as f:
        json.dump(args, f)
