

def train(
        sites,
        models,
        predictors
):
    """
    Train models using predictors on sitees.

    Args:
        sites (str): Comma-separated list of site IDs to train on.
                     Must match the name(s) of the data directories.
        models (str): Comma-separated list of model names to train.
                      Options: ['rf', 'ann', 'lasso', 'xgb']
        predictors (str): Either a comma-separated list of predictors
                          or a path to a file with predictor names.
                          See train/predictors.json for an example.

    Saves trained models to data/{SiteID}/{model}/{predictor}/
    for each {SiteID}, {model}, and {predictor} subset.
    """
    data_dir = Path("data/")
    sites = sites.split(",")
    models = models.split(",")
    for site in sites:
        site_data_dir = data_dir / site
        for predictor in predictors:
            # TODO: If train/valid/test csvs don't exist, tell user to run preprocessing
            # TODO: Add data munging for input to ML here
            for model in models:
                print(f"Training model {model} for site {site}...")
                model_dir = site_data_dir / model / predictor
                model_dir.mkdir(exist_ok=True)
                for i in range(n_train):
                    model_path = model_dir / f"model{i+1}.pkl"


                    model_obj.save(model_path)
