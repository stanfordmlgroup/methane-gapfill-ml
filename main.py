import fire

from preprocess import preprocess
from train import train
from test import test
from gapfill import gapfill

def run_all(sites, models, predictors, **kwargs):
    preprocess(sites, **kwargs)
    train(sites, models, predictors, **kwargs)
    test(sites, models, predictors, **kwargs)
    gapfill(sites, models, predictors, **kwargs)


if __name__ == "__main__":
    fire.Fire()
