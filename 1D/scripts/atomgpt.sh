#!/bin/bash
set -e
mkdir -p atomgpt_ehull
uv pip install pymatgen pandas tqdm
python scripts/preprocess.py atomgpt \
    --input interim/* \
    --output atomgpt_ehull \
    --target e_above_hull \
    --seed 1738
