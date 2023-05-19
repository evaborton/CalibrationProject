# Calibration Project

Creates models on a Covid-19 fake news dataset and calibrates them.

### Dependencies

The following packages must be installed:
- transformers
- datasets
- evaluate
- uncertainty-calibration (use pip not conda)
- tensorflow
- tensorflow-probability

Note: Installing tensorflow-probability with conda requires
```
conda install -c conda-forge tensorflow-probability
```

### Usage

#### The dataset

Raw data is in the data folder, split into training, validation, and test sets.

#### Building the models

Code for building the models is in ModelBuilder.ipynb.

#### Calibration

Code for calibrating the models is in Calibration.ipynb.
