import numpy as np
import pickle as pkl
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV

from metrics import metric_dict


class BaseModel(object):
    """Base class for all models."""

    def __init__(self, cv=5, n_iter=20):
        self.cv = cv
        self.n_iter = n_iter

    def fit(self, X, y):
        """Train on a training set and select optimal hyperparameters."""
        if X.isna().any().any():
            self.imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            X.loc[:, :] = self.imputer.fit_transform(X)
        else:
            self.imputer = None

        if self.scaler is not None:
            X.loc[:, :] = self.scaler.fit_transform(X)
        random_search = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=self.param_dist,
            n_iter=self.n_iter,
            cv=self.cv,
            scoring="neg_mean_squared_error",
            n_jobs=-1
        )

        random_search.fit(X, y)
        self.model = random_search.best_estimator_
        self.predictors = X.columns.tolist()

    def predict(self, X):
        if self.imputer is not None:
            X.loc[:, :] = self.imputer.transform(X)
        if self.scaler is not None:
            X.loc[:, :] = self.scaler.transform(X)
        return self.model.predict(X)

    def evaluate(self, X, y, metric):
        y_hat = self.model.predict(X)
        if metric not in metric_dict:
            raise ValueError(f"Metric {metric} not supported.")
        metric_fn = metric_dict[metric]
        return metric_fn(y, y_hat)

    def save(self, path):
        """Save model to path."""
        with open(path, 'wb') as f:
            pkl.dump(self, f)

    @property
    def feature_importances(self):
        weights = self.model.feature_importances_
        return dict(sorted(zip(
            self.predictors, weights),
            key=lambda x: x[-1],
            reverse=True
        ))
