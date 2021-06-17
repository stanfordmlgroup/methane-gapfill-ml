import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV

from predictors import *
from metrics import metric_dict


class BaseModel(object):
    """Base class for all models."""

    def __init__(self, predictor_subset, cv=5, n_iter=20):
        self.predictor_subset = predictor_subset
        self.cv = cv
        self.n_iter = n_iter

    def preprocess(self, X):
        """Prepare X to be input to the model."""
        predictor_subset = self.predictor_subset.copy()
        if 'all' in predictor_subset:
            predictor_subset = add_all_predictors(predictor_subset, X.columns)
        
        use_temporal = 'temporal' in predictor_subset
        if use_temporal:
            X_temporal = get_temporal_predictors(
                X['TIMESTAMP_END']
            )
            predictor_subset.remove('temporal')

        X = X[predictor_subset]

        if use_temporal:
            X = pd.concat([X, X_temporal], axis=1)

        if 'WD' in predictor_subset:
            X = process_wind_direction_predictor(X)

        return X

    def impute(self, X):
        """Impute missing predictors."""
        if X.isna().any().any():
            self.imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            X.loc[:, :] = self.imputer.fit_transform(X)
        else:
            self.imputer = None
        return X

    def fit(self, X, y):
        """Train on a training set and select optimal hyperparameters."""
        X = self.preprocess(X)
        X = self.impute(X)

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
        X = self.preprocess(X)
        if self.imputer is not None:
            X.loc[:, :] = self.imputer.transform(X)
        if self.scaler is not None:
            X.loc[:, :] = self.scaler.transform(X)
        return self.model.predict(X)

    def evaluate(self, X, y, metric):
        y_hat = self.predict(X)
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
