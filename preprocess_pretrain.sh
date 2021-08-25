#!/bin/bash
mkdir data_processing
cd preprocessing_scripts
git clone https://github.com/AnReu/ARQMathCode.git
python3 get_clean_json.py
python3 get_pretraining_data_base.py
python3 get_pretraining_data_separated.py