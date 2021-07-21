# methane-gapfill-ml
Python codebase for [our manuscript](https://authors.elsevier.com/a/1dNxrcFXJZ1gC) "Gap-filling eddy covariance methane fluxes: Comparison of machine learning model predictions and uncertainties at FLUXNET-CH4 wetlands".

This codebase is a work in progress and will be updated periodically over the next month:
- [X] Data preprocessing
- [X] Model training
- [X] Model testing
- [X] Uncertainty estimation, calibration/sharpness, and post-processing calibration
- [ ] Gap-filling a dataset with a trained model

## Prerequisites
1. Clone this repository
```Shell
git clone git@github.com:stanfordmlgroup/methane-gapfill-ml.git
```

2. Make the virtual environment:

```Shell
conda env create -f environment.yml
```

3. Activate the virtual environment:

```Shell
source activate ch4-gap-ml
```

## Usage 
Prepare data in a CSV and include the following headers:
- `TIMESTAMP_END`: Format YYYYMMDDHHmm (e.g. 201312060030)
- `Year`: Four digit integer year (e.g. 2013)
- `DOY`: Integer day between 1 and 365 (e.g. 217, 340)
- `Hour`: Decimal hour between 0.0 and 23.5 (e.g. 0.5, 10.0)
- `FCH4`: Methane flux in nmol m<sup>-2</sup> s<sup>-1</sup>
All other headers will be treated as input predictors.

Create a folder called `data/` and make another folder in `data/`
using the site ID (`data/{SiteID}/`). This is where all of the processed
data for the site will be written.

Place the CSV in this folder and name it `raw.csv`, so the full path to the
CSV should be `data/{SiteID}/raw.csv`, where `{SiteID}` should be replaced
with the actual ID of the site.

Preprocess the data
```Shell
python main.py preprocess
```

Train models
```Shell
python main.py train
```

Evaluate a trained model
```Shell
python main.py test
```

Gapfill using a trained model (not yet implemented)
```Shell
python main.py gapfill
```

Run all steps, including data preprocessing, model training, model evaluation, and gapfilling
```Shell
python main.py run_all
```

Run `python main.py {preprocess,train,test,gapfill} --help` for descriptions of all of the command-line arguments.

Example commands using the sample data in the repository:
```Shell
python main.py preprocess --sites NZKop --eval_frac 0.1 --n_train 10
python main.py train --sites NZKop --models [lasso,rf] --predictors_paths train/predictors.txt
```
When specifying multiple values for a parameter, you can either use a comma-separated string or list syntax like in the above command.

## Contributions
This tool was developed by Jeremy Irvin, Yulun Zhou, Fred Lu, Vincent Liu, and Sharon Zhou.

## License

[Apache License 2.0](https://github.com/stanfordmlgroup/methane-gapfill-ml/blob/master/LICENSE).

## Citing
If you're using this codebase, please cite [this paper](https://authors.elsevier.com/a/1dNxrcFXJZ1gC).
