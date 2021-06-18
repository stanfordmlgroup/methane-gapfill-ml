import pickle
import pandas as pd
from pathlib import Path
from collections import defaultdict

from models import EnsembleModel
from metrics import metric_dict


def test(
        model_dirs=None,
        sites=None,
        models=None,
        predictors=None,
        predictor_paths=None,
        split='test',
        eval_metrics=metric_dict.keys()
):
    """
    Args:
        model_dirs (list<str>): Comma-separated list of paths to model
                                directories with checkpoints. These must all
                                be from the same site.
        split (str): Which split to test on. Options: ['train', 'valid', 'test']
        eval_metrics (list<str>): Metrics to use to evaluate the model(s) on
                                  the split.

    Writes performance metrics to {model_dir}/{split}_results/
    for each {model_dir}.
    """
    if isinstance(model_dirs, str):
        model_dirs = model_dirs.split(",")
    # # TODO: Add logic for when user passes in sites/models/predictors instead of paths
    # # (just triple for loop looking for models, print when didn't find one)
    # if model_dirs is None:
    #     model_dirs = []
    #     for model

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
