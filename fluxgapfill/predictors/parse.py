from pathlib import Path


def parse_predictors(predictors_paths, predictors):
    """Parse the predictors supplied through the command-line.

    Args:
        predictors_paths (list<str>): Comma-separated list path file(s) with
                                      predictor names. See
                                      predictors/metereological.txt for an
                                      example.
        predictors (list<str>): Comma-separated list of predictors. Ignored if
                                predictors_path is provided.
                                Certain keyword predictors are used to denote
                                specific sets of predictors:
                                ['temporal', 'all']

    Return a dictionary mapping the name of each predictor subset to a list of
    predictors.
    """
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
   
    return predictor_subsets
