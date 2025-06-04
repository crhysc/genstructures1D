#!/bin/bash
mkdir -p ../models/cdvae/data/supercon/
uv pip install pymatgen pandas tqdm
python scripts/preprocess.py cdvae \
	--input interim/* \
	--output . \
	--target e_above_hull \
	--seed 1738
wget "https://raw.githubusercontent.com/crhysc/utilities/refs/heads/main/supercon.yaml"
python - <<'PYCODE'
import os
path = "../models/cdvae/data/supercon"
files = ["train.csv", "test.csv", "val.csv"]
for file in files:
	file_path = os.path.join(path,file)
	if os.path.exists(file_path):
		os.remove(file_path)
	os.rename(file, file_path)
yaml_path = "../models/cdvae/conf/data/supercon.yaml"
if not os.path.exists(yaml_path):
	os.rename("supercon.yaml", yaml_path)
else:
	os.remove("supercon.yaml")
PYCODE

