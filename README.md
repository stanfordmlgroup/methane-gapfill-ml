# methane-gapfill-ml
Python codebase for [our manuscript](https://authors.elsevier.com/a/1dNxrcFXJZ1gC) "Gap-filling eddy covariance methane fluxes: Comparison of machine learning model predictions and uncertainties at FLUXNET-CH4 wetlands".

This codebase implements the following features:
- [X] Data preprocessing
- [X] Model training
- [X] Model testing
- [X] Uncertainty estimation, calibration/sharpness, and post-processing calibration
- [X] Gap-filling a dataset with a trained model



## Jupyter Notebook Tutorial
https://github.com/stanfordmlgroup/methane-gapfill-ml/blob/pip-package/fluxgapfill_tutorial.ipynb


## Installation
```Shell
pip install gapfluxfill
```


## Usage 
Prepare data in a CSV and include the following headers:
- `TIMESTAMP_END`: Format YYYYMMDDHHmm (e.g. 201312060030)
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
python main.py train --sites NZKop --models [lasso,rf] --predictors_paths predictors/meteorlogical.txt
```
When specifying multiple values for a parameter, you can either use a comma-separated string or list syntax like in the above command.


## Contributions
This tool was developed by Jeremy Irvin, Yulun Zhou, Fred Lu, Vincent Liu, and Sharon Zhou.

## License

[Apache License 2.0](https://github.com/stanfordmlgroup/methane-gapfill-ml/blob/master/LICENSE).

## Citing
If you're using this codebase, please cite:

1) The Gapfilling Algorithm: 
Irvin, J., Zhou, S., McNicol, G., Lu, F., Liu, V., Fluet-Chouinard, E., ... &amp; Jackson, R. B. (2021). Gap-filling eddy covariance methane fluxes: Comparison of machine learning model predictions and uncertainties at FLUXNET-CH4 wetlands. Agricultural and Forest Meteorology, 308, 108528.

2) The Python-Toolkit:
In text: "We used the FluxGapfill python toolkit (Version 0.2.0; Irvin, et al, 2021) to complete our work."


References

Irvin, J., Zhou, Y., McNicol, G., &amp; Liu, J. (2021). FluxGapfill: A Python Interface for Machine-learning Driven Methane Gap-filling. Version 0.2.0. Zenodo. https://doi.org/10.5281/zenodo.5515761. Accessed 2021-09-19.
