

def check_predictors_present(data_set, predictor_subset):
    """Check if all predictors in predictor_subset are present
    as columns in the data_set dataframe."""
    key_word_predictors = ['temporal', 'all']

    for predictor in predictor_subset:
        if (
                predictor not in data_set.columns and
                predictor not in key_word_predictors
        ):
            raise ValueError(f"Predictor {predictor} not found " +
                                "in the data.")
