# methane-gapfill-ml
Python codebase for our paper "Gap-filling eddy covariance methane fluxes: Comparison of machine learning model predictions and uncertainties at FLUXNET-CH4 wetlands".

Read more about our project [here](https://stanfordmlgroup.github.io/projects/gapfill/) and our manuscript [here](https://add/link/once/published).

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

Gapfill using a trained model
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
python main.py train --sites NZKop --models [lasso,rf] --predictors_path train/predictors.txt
```

## Contributions
This tool was developed by Jeremy Irvin, Fred Lu, Vincent Liu, and Sharon Zhou.

## Citing
If you're using this codebase, please cite [this paper](https://add/link/once/published):
