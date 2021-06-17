import fire

from preprocess import preprocess
from train import train
from test import test
from gapfill import gapfill

def run_all(**kwargs):
    preprocess(**kwargs)
    train(**kwargs)
    test(**kwargs)
    gapfill(**kwargs)


if __name__ == "__main__":
    fire.Fire()
