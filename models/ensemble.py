import numpy as np
import pickle as pkl
from collections import defaultdict

from metrics import metric_dict


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

            self.split2model = lambda:0
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

    def predict(self, X):
        model_preds = []
        for split, model in self.split2model.items():
            model_preds.append(model.predict(X))

        ensemble_pred = np.mean(model_preds, axis=0)
        return ensemble_pred

    def evaluate(self, X, y, metric):
        y_hat = self.predict(X)
        if metric not in metric_dict:
            raise ValueError(f"Metric {metric} not supported.")
        metric_fn = metric_dict[metric]
        return metric_fn(y, y_hat)
