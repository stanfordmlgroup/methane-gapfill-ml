import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from scipy.stats import pearsonr


def pearson_r_squared(truth, prediction):
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


def get_pred_interval(pred_dist):
    return np.array([dist.dist.interval(0.95) for dist in pred_dist])


def calibration(truth, pred_dist):
    pred_interval = get_pred_interval(pred_dist)
    frac_of_truth_in_interval = (
        (truth > pred_interval[:, 0]) &
        (truth < pred_interval[:, 1])
    ).mean()
    return frac_of_truth_in_interval


def sharpness(truth, pred_dist):
    pred_interval = get_pred_interval(pred_dist)
    widths = np.diff(pred_interval, axis=1)
    return widths.mean()


def normalized_sharpness(truth, pred_dist):
    pred_interval = get_pred_interval(pred_dist)
    widths = np.diff(pred_interval, axis=1)
    return widths.mean() / np.std(truth)


uncertainty_metric_dict = {
    "calibration": calibration,
    "sharpness": sharpness,
    "normalized_sharpness": normalized_sharpness
}
