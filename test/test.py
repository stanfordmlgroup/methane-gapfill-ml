import pickle
import pandas as pd
from pathlib import Path
from collections import defaultdict

from models import EnsembleModel
from metrics import metric_dict
from predictors import parse_predictors


def test(
        model_dirs=None,
        sites=None,
        models=None,
        predictors=None,
        predictors_paths=None,
        split='test',
        eval_metrics=metric_dict.keys()
):
    """
    Evaluate models.

    Args:
        model_dirs (list<str>): Comma-separated list of paths to model
                                directories with checkpoints. These must all
                                be from the same site. sites, models, predictors,
                                and predictors_paths will be ignored if this
                                parameter is supplied to the function.
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
        split (str): Which split to test on. Options: ['train', 'valid', 'test']
        eval_metrics (list<str>): Metrics to use to evaluate the model(s) on
                                  the split.

    Writes performance metrics to data/{site}/{split}_results/
    """
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
                for predictor_subset in predictor_subsets:
                    model_dir = save_dir / model / predictor_subset
                    if len(list(model_dir.glob("*.pkl"))) == 0:
                        raise ValueError("No models trained for " + 
                                         f"site={site}, " +
                                         f"model={model}, " +
                                         f"predictors={predictor_subset}, "
                                         )
                    model_dirs.append(model_dir)


    splits = ['train', 'valid', 'test']
    if split not in splits:
        raise ValueError(f"Got split={split} but must be one of " +
                         f"{','.join(splits)}")

    site_data_dirs = set(
        Path(model_dir).parent.parent.parent
        for model_dir in model_dirs
    )
    if len(site_data_dirs) != 1:
        raise ValueError('Model paths correspond to more than one site.')

    site_data_dir = site_data_dirs.pop()

    eval_scores = []
    for model_dir in model_dirs:
        model_dir = Path(model_dir)

        predictor_subset = model_dir.name
        model = model_dir.parent.name

        if split == 'test':
            eval_df = pd.read_csv(site_data_dir / 'test.csv')
            model_obj = EnsembleModel(model_dir)
            scores = {}
            for eval_metric in eval_metrics:
                score = model_obj.evaluate(eval_df, eval_df['FCH4'], eval_metric)
                scores[eval_metric] = [score]

            scores['model'] = model
            scores['predictors'] = predictor_subset
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
                    raise ValueError(f"Model path {model_path} does not exist.")
                with model_path.open('rb') as f:
                    model_obj = pickle.load(f)
                
                for eval_metric in eval_metrics:
                    score = model_obj.evaluate(eval_df, eval_df['FCH4'], eval_metric)
                    scores[eval_metric].append(score)
            
            scores_df = pd.DataFrame(scores)
            mean_scores = scores_df.mean().to_frame().T
            mean_scores['model'] = model
            mean_scores['predictors'] = predictor_subset
            eval_scores.append(mean_scores)
    
    eval_scores_df = pd.concat(eval_scores)
    eval_scores_path = site_data_dir / f'{split}_results.csv'
    print(f"Writing {split} metrics to {eval_scores_path}")
    eval_scores_df.to_csv(eval_scores_path, index=False)
