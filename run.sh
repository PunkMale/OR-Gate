#!/bin/bash

# source activate
# conda deactivate
# set -e
# set -u

# conda activate pytorch

chmod 777 main.py
chmod 777 parser.py

#nohup python main.py > logs/v1_r=0.0_w=5_k=90.log 2>&1 &
python main.py
