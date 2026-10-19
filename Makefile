# Pipeline entry points. `make all` runs everything with the from-scratch code;
# `make all BACKEND=reference` uses StepMix/SciPy/scikit-learn instead.

PY      ?= uv run python
BACKEND ?= ours
METHOD  ?= hierarchical
NBOOT   ?= 200
STARTS  ?= 20

.PHONY: all data fit checks describe forward overview test lint clean

all: data fit checks describe forward

data: data/processed/analysis.parquet

data/processed/analysis.parquet: config/conditions.yaml scripts/01_select.py scripts/02_encode.py
	$(PY) scripts/01_select.py --download
	$(PY) scripts/02_encode.py

fit: data
	$(PY) scripts/03_fit_lca.py --backend $(BACKEND) --n-starts $(STARTS) --permutation-null

checks: fit
	$(PY) scripts/04_crosscheck.py --backend $(BACKEND) --method $(METHOD)
	$(PY) scripts/05_replicate.py --backend $(BACKEND) --n-starts $(STARTS)
	$(PY) scripts/06_bootstrap.py --backend $(BACKEND) --n-boot $(NBOOT) --n-starts $(STARTS)

describe: fit
	$(PY) scripts/07_describe.py

forward: fit
	$(PY) scripts/08_forward_2025.py --backend $(BACKEND) --n-starts $(STARTS)

overview: data
	$(PY) scripts/make_data_overview.py

test:
	uv run pytest -q

lint:
	uv run ruff check . && uv run ruff format --check .

clean:
	rm -rf results/*.json results/*.csv results/*.npz results/*.npy results/figures
