#!/bin/bash
set -e
mkdir -p processed_agpt
pip install -r data_reqs.txt
python generate_id_prop.py \
    -i interim
    -o processed_agpt
