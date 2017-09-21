This repo aims at automatically optimizing the rmc map matcher's hyperparameters.
The work is part of my masterthesis.

Dependencies:
* python3
* numpy
* scipy
* matplotlib
* scikit-learn
* bayesian-optimization (https://github.com/fmfn/BayesianOptimization)
* rmc\_gbr\_mapping on branch evalualtion\_pipeline\_tweaks

Usage:
The main script is experiment\_coordinator.py, but please use wrapper.sh to run it.
It's a simple wrapper script which sets PYTHONHASHSEED to a fixed value.
Without it, the SampleDatabase won't be able to recognize previously generated Samples.
