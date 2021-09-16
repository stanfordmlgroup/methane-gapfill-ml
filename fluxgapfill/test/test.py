import os
import json
import pickle
import pandas as pd
from pathlib import Path
from collections import defaultdict

from fluxgapfill.models import EnsembleModel
from fluxgapfill.metrics import metric_dict, uncertainty_metric_dict
from fluxgapfill.predictors import parse_predictors
from fluxgapfill.models import get_model_class

def parse_model_dirs(model_dirs, sites, models, predictors,
                     predictors_paths, data_dir):
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
    return model_dirs


def test(
        data_dir, 
        sites=None,
        models=None,
        predictors=None,
        predictors_paths=None,
        split='test',
        distribution="laplace",
        eval_metrics=list(metric_dict.keys()),
        uncertainty_eval_metrics=list(uncertainty_metric_dict.keys()),
        overwrite_results=False,
        **kwargs
):
    """
    Evaluate models.

    Args:
        data_dir (<str>): path to the project folder containing site data foloders to save
                          all raw, processed data and results. 
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
                                      *if both predictors and predictor_paths set to
                                      None, all predictor_subsets with trained models
                                      will be tested. j
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

    if (predictors is not None) and (predictors_paths is not None):
        predictor_subsets = parse_predictors(predictors_paths, predictors)
    else:
        predictor_subsets = None # process all predictor_subsets trained
    
    if sites is None:
        sites = os.listdir(data_dir)
        if "results" in sites:
            sites.remove("results")
    
    ##
    # model_dirs = parse_model_dirs(
    #     model_dirs=model_dirs,
    #     sites=sites,
    #     models=models,
    #     predictors=predictors,
    #     predictors_paths=predictors_paths,
    #     data_dir=data_dir
    # )


    
    splits = ['train', 'valid', 'test']
    if split not in splits:
        raise ValueError(f"Got split={split} but must be one of " +
                         f"{','.join(splits)}")
    
    eval_scores_dir = data_dir/"results"
    eval_scores_dir.mkdir(exist_ok=True)

    eval_scores_fn = eval_scores_dir / f"{split}.csv"
    if eval_scores_fn.exists() and not overwrite_results:
        raise ValueError(f"Results path at {eval_scores_fn} already exists. Consider set overwrite_results to True.")

    eval_scores = []
    for site in sites:
        site_data_dir = data_dir/site
        site_models_dir = site_data_dir/"models"
        if models is None:
            models = os.listdir(site_models_dir)
        
        for model in models:
            site_model_dir = site_models_dir/model
            if predictor_subsets is None:
                predictor_subsets = os.listdir(site_model_dir) ##
            
            for predictor_subset in predictor_subsets:
                print('Model testing...\n' +
                      f" - site: {site}\n" +
                      f" - model: {model}\n" +
                      f" - predictors: {predictor_subset}")

                site_model_predictor_dir = site_model_dir/predictor_subset
                
                if split == 'test':
                    eval_df = pd.read_csv(site_data_dir/'test.csv')
                    Model = EnsembleModel(site_model_predictor_dir)
                    y = eval_df['FCH4']
                    y_hat_dist = Model.predict_dist(
                        eval_df,
                        distribution=distribution
                    )
                    y_hat = [dist.mean() for dist in y_hat_dist]
                    y_hat_scale = [dist.scale for dist in y_hat_dist]
                    uncertainty_scale = Model.uncertainty_scale(
                        y, y_hat_dist, distribution=distribution
                    )
                    y_hat_dist_scaled = Model.predict_dist(
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
                    
                    pred_df.to_csv(site_model_predictor_dir / f"{split}_predictions.csv", index=False)
                    
                    scale_path = site_model_predictor_dir / "scale.json"
                    with scale_path.open('w') as f:
                        json.dump(uncertainty_scale, f)
        
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
                    
                    # format output
                    scores['site'] = site
                    scores['model'] = model
                    scores['predictors_subset'] = predictor_subset
                    predictors = [
                        predictor
                        for predictor in Model.predictors
                        if predictor != "intercept"
                    ]
                    scores['predictors'] = ";".join(predictors)
                    model_scores = pd.DataFrame(scores)
                    model_scores_fn = site_model_predictor_dir / f"{split}_results.csv"
                    print(f" - Writing {split} metrics to: {model_scores_fn}\n")
                    model_scores.to_csv(model_scores_fn, index=False)
                    eval_scores.append(model_scores)
        
                else:
                    # Run each model on the corresponding data split
                    site_training_data_dir = site_data_dir / 'training'
                    n_train = len(list(site_training_data_dir.glob("train*.csv")))
        
                    scores = defaultdict(list)
                    for i in range(n_train):
                        eval_fn = site_training_data_dir / f'{split}{i+1}.csv'
                        eval_df = pd.read_csv(eval_fn)
        
                        model_fn = site_model_predictor_dir / f"model{i+1}.pkl"
                        if not model_fn.exists():
                            raise ValueError(
                                f"Model path {model_fn} does not exist."
                            )
                        with open(model_fn, 'rb') as f:
                            Model = pickle.load(f)
        
                        y_hat = Model.predict(eval_df)
                        y = eval_df['FCH4']
                        pred_df = pd.DataFrame({"groundtruth": y, "prediction": y_hat})
                        pred_df.to_csv(
                            site_model_predictor_dir / f"{split}{i+1}_predictions.csv",
                            index=False
                        )
                        scores = {
                            eval_metric: [metric_dict[eval_metric](y, y_hat)]
                            for eval_metric in eval_metrics
                        }
                        
                    # format output
                    scores_df = pd.DataFrame(scores)
                    mean_scores['site'] = site
                    mean_scores = scores_df.mean().to_frame().T
                    mean_scores['model'] = model
                    mean_scores['predictor_subset'] = predictor_subset
                    predictors = [
                        predictor
                        for predictor in Model.predictors
                        if predictor != "intercept"
                    ]
                    mean_scores['predictors'] = ";".join(predictors)
                    mean_scores_fn = site_model_predictor_dir / f"{split}_results.csv"
                    print(f"Writing {split} metrics\n" +
                          f" - site: {site}\n" +
                          f" - model: {model}\n" +
                          f" - predictor: {predictor_subset}\n" +
                          f" - to: {mean_scores_fn}\n")
                    mean_scores.to_csv(mean_scores_fn, index=False)
                    eval_scores.append(mean_scores)

                eval_scores_df = pd.concat(eval_scores)
                eval_scores_df.to_csv(eval_scores_fn, index=False)

    # eval_scores_df = pd.concat(eval_scores)
    # print(f"Writing aggregated {split} metrics to {eval_scores_fn}")
    # eval_scores_df.to_csv(eval_scores_fn, index=False)

    # Save args to eval_scores_dir
    eval_args_fn = eval_scores_dir / f"{split}_args.json"
    with open(eval_args_fn, 'w') as f:
        json.dump(args, f)
