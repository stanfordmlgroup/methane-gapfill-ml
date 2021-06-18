import numpy as np
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error,
    r2_score
)
from scipy.stats import pearsonr


def pearson_r_squared(truth, prediction):
    """ pearson r^2 score on truth and prediction arrays """
    return pearsonr(truth, prediction)[0] ** 2


def reference_standard_dev(truth, prediction):
    return np.std(truth)


def normalized_mean_absolute_error(truth, prediction):
    return (mean_absolute_error(truth, prediction) /
            reference_standard_dev(truth, prediction))


def bias(truth, prediction):
    return (prediction - truth).mean()


metric_dict = {
    "mse": mean_squared_error,
    "mae": mean_absolute_error,
    "nmae": normalized_mean_absolute_error,
    "r2": r2_score,
    "pr2": pearson_r_squared,
    "bias": bias
}
