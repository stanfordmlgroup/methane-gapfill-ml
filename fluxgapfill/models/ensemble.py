import numpy as np
import pickle as pkl
from collections import defaultdict
from ngboost.distns.normal import Normal
from ngboost.distns.laplace import Laplace

from fluxgapfill.metrics import metric_dict


class EnsembleModel(object):
    def __init__(self, model_dir, use_iterator=False):
        self.model_dir = model_dir
        self.model_name = model_dir.parent.name
        self.use_iterator = use_iterator
        # Load all models from directory.
        if use_iterator:
            def _iterator():
                for model_path in self.model_dir.glob("*.pkl"):
                    with open(model_path, 'rb') as f:
                        yield model_path.stem, pkl.load(f)

            self.split2model = lambda: 0
            self.split2model.items = _iterator

        else:
            self.split2model = {}
            for model_path in self.model_dir.glob("*.pkl"):
                with open(model_path, 'rb') as f:
                    model = pkl.load(f)
                self.split2model[model_path.stem] = model
            # Assume every model in the ensemble has the same set of predictors
            self._predictors = model.predictors

    @property
    def predictors(self):
        if self.use_iterator:
            for _, model in self.split2model.items():
                return model.predictors
        else:
            return self._predictors

    @property
    def feature_importances(self):
        n_models = len(self.split2model)
        feature_importances = defaultdict(float)
        for _, model in self.split2model.items():
            model_feature_importances = model.feature_importances
            for var, value in model_feature_importances.items():
                feature_importances[var] += value

        for var in feature_importances:
            feature_importances[var] /= n_models
        return dict(feature_importances)

    def predict_individual(self, X):
        """Return individual model predictions on inputs X"""
        model_preds = []
        for split, model in self.split2model.items():
            model_preds.append(model.predict(X))
        return np.array(model_preds)

    def predict(self, X):
        """Predict the mean of the ensemble model predictions on inputs X"""
        model_preds = self.predict_individual(X)

        ensemble_pred = np.mean(model_preds, axis=0)
        return ensemble_pred

    def predict_dist(self, X, distribution="laplace",
                     uncertainty_scale=None):
        """Predict a distribution parametrized by the mean and variance
        of the ensemble model predictions on inputs X.

        X (pandas DataFrame or numpy array): Inputs for prediction
        distribution (str): Predicted distribution to parameterize.
                            Options: [laplace, normal]
        uncertainty_scale (float): Optional platt scale value to scale
                                   uncertainties.
        """
        model_preds = self.predict_individual(X)

        ensemble_mean = np.mean(model_preds, axis=0)
        ensemble_var = np.var(model_preds, axis=0)

        if distribution == "laplace":
            Dist = Laplace
            if uncertainty_scale is None:
                ensemble_scale = np.sqrt(ensemble_var) / 2
            else:
                ensemble_scale = (
                    np.sqrt(ensemble_var) / 2 * uncertainty_scale
                )

        elif distribution == "normal":
            Dist = Normal
            if uncertainty_scale is None:
                ensemble_scale = np.sqrt(ensemble_var)
            else:
                ensemble_scale = np.sqrt(ensemble_var * uncertainty_scale)

        else:
            raise ValueError(f"Distribution {distribution} not supported.")

        ensemble_dist = Dist(
            np.array([ensemble_mean, np.log(ensemble_scale)])
        )

        return [dist for dist in ensemble_dist]

    def evaluate(self, X, y, metric):
        y_hat = self.predict(X)
        if metric not in metric_dict:
            raise ValueError(f"Metric {metric} not supported.")
        metric_fn = metric_dict[metric]
        return metric_fn(y, y_hat)

    def uncertainty_scale(self, truth, pred_dist, distribution="laplace"):
        """Compute MLE scaling factor (platt scaling)"""
        mean = np.array([dist.mean() for dist in pred_dist])
        scale = np.array([dist.scale for dist in pred_dist])
        if distribution == "laplace":
            numerator = np.absolute(truth.values - mean)
            denominator = scale
            return (numerator / denominator).mean()
        elif distribution == "normal":
            numerator = (truth.values - mean) ** 2
            denominator = scale ** 2
            return (numerator / denominator).mean()
        else:
            raise ValueError(f"Distribution {distribution} not supported.")
