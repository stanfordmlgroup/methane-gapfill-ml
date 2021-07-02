import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.preprocessing import StandardScaler

from .base import BaseModel


class Lasso(BaseModel):
    """Class for Lasso model."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = LassoCV(
            eps=0.001,
            n_alphas=100,
            max_iter=10000,
            tol=1e-4,
            cv=self.cv
        )
        self.scaler = StandardScaler()

    def fit(self, X, y):
        X = self.preprocess(X)
        X = self.impute(X)

        X.loc[:, :] = self.scaler.fit_transform(X)

        self.model.fit(X, y)
        self.predictors = X.columns.tolist() + ['intercept']
        self.model.feature_importances_ = np.concatenate(
            (self.model.coef_, np.expand_dims(self.model.intercept_, 0))
        )


class ANN(BaseModel):
    """Class for Artificial Neural Network model."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = MLPRegressor()
        self.param_dist = {
            "hidden_layer_sizes": [(i, ) for i in range(5, 30)] +
                                  [(i, int(i * 0.5)) for i in range(5, 30)] +
                                  [(i, int(i * 0.75)) for i in range(5, 30)],
            "solver": ["lbfgs", "adam"],
            "learning_rate_init": [0.01, 0.001, 0.0001],
            "max_iter": [100000],
            "tol": [1e-7],
            "activation": ["tanh", "relu"]
        }

    @property
    def feature_importances(self):
        # MLP has no intrinsic feature importance values.
        self.model.feature_importances_ = [
            -float("inf") for _ in self.predictors
        ]


class RandomForest(BaseModel):
    """Class for Random Forest model."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = RandomForestRegressor(n_jobs=-1)
        self.param_dist = {
            "n_estimators": np.linspace(start=50, stop=500, num=10).astype(int),
            "max_features": ['auto', 'sqrt'],
            "max_depth": list(np.linspace(10, 110, num=11).astype(int)) + [None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "bootstrap": [True, False]
        }
        self.scaler = None


class XGBoost(BaseModel):
    """Class for XGBoost model."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = XGBRegressor()
        self.param_dist = {
            "n_estimators": np.linspace(start=50, stop=500, num=10).astype(int),
            "max_depth": np.linspace(10, 110, num=11).astype(int),
            "min_child_weight": [2, 5, 10],
            "subsample": [i / 100.0 for i in range(75, 100, 10)],
            "gamma": [i / 10.0 for i in range(0, 5, 2)],
            "colsample_bytree": [i / 10.0 for i in range(6, 10)],
            "objective": ["reg:squarederror"],
            "booster": ["gbtree"],
            "n_jobs": [-1],
            "learning_rate": [0.1],
            "scale_pos_weight": [1]
        }
        self.scaler = None


def get_model_class(model):
    model_dict = {
        "lasso": Lasso,
        "ann": ANN,
        "rf": RandomForest,
        "xgb": XGBoost
    }
    return model_dict[model]
