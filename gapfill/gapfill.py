

def gapfill(
        sites,
        models,
        predictors
):
    """
    Args:
        sites (str): Comma-separated list of site IDs to train on.
                     Must match the name(s) of the data directories.
        models (str): Comma-separated list of model names to train.
                      Options: ['rf', 'ann', 'lasso', 'xgb']
        predictors (str): Either a comma-separated list of predictors
                          or a path to a file with predictor names.
                          See train/predictors.json for an example.

    Writes gapfilled data to data/{SiteID}/gapfilled.csv
    where gapfilled.csv has all the same columns as raw.csv
    plus extra columns for each of the models and predictor subsets.
    """
    pass