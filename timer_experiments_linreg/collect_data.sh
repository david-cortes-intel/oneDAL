#!/bin/env zsh
source ~/get_onedal.sh
export PYTHONPATH=~/repos/scikit-learn-intelex/
mamba activate py312

python collect_combinatorial_data.py
python collect_mean_variance_data.py
python collect_variance_multiparams.py
python collect_sampling_strategy_data.py
python collect_pipeline_times_data.py
