# Pipeline entry points. Run `make help` for a list.
#
# BACKEND=reference uses StepMix / SciPy / scikit-learn and works today.
# BACKEND=ours uses the from-scratch code in src/nsch_subtypes/ once it is
# implemented (see docs/implementation-spec.md); until then it stops at step 3.

PY      ?= uv run python
BACKEND ?= reference
STARTS  ?= 20
NBOOT   ?= 200
KS      ?= 4 5 6 7

.PHONY: help all download fit sweep choose calibrate figures overview test smoke lint clean

help:
	@echo "make test       unit tests on synthetic data (seconds)"
	@echo "make smoke      run every pipeline step once at tiny settings (about 5 minutes)"
	@echo "make fit        step 3: latent class models K=1..8 into results/"
	@echo "make sweep      steps 4 to 14 at K in KS ($(KS)) and the figures (hours; mostly bootstrap)"
	@echo "make choose     step 13 only: apply the K rule to the bootstraps on file"
	@echo "make calibrate  step 14 only: shape and stability checks by simulation (about an hour)"
	@echo "make figures    redraw the results figures from existing outputs"
	@echo "make overview   the data-overview figures (needs data/interim from 'make download')"
	@echo "make download   steps 1 and 2 from the Census files (about 60 MB; not needed for anything else)"
	@echo "make all        fit + sweep"
	@echo "Options: BACKEND=$(BACKEND) STARTS=$(STARTS) NBOOT=$(NBOOT) KS='$(KS)'"

all: fit sweep

# data/processed/analysis.parquet and codebook.csv are committed, so nothing
# below needs the download. Rebuild them on purpose with `make download`.
download:
	$(PY) scripts/01_select.py --download
	$(PY) scripts/02_encode.py

fit:
	$(PY) scripts/03_fit_lca.py --backend $(BACKEND) --n-starts $(STARTS) --permutation-null

sweep:
	BACKEND=$(BACKEND) STARTS=$(STARTS) NBOOT=$(NBOOT) bash scripts/sweep_k.sh $(KS)

choose:
	$(PY) scripts/13_heldout_selection.py --backend $(BACKEND) --reuse-heldout

calibrate:
	$(PY) scripts/14_shape_and_stability.py --backend $(BACKEND)

figures:
	$(PY) scripts/make_results_figures.py

overview:
	$(PY) scripts/make_data_overview.py

test:
	uv run pytest -q -m "not slow"

smoke:
	uv run pytest -q -m slow

lint:
	uv run ruff check . && uv run ruff format --check .

clean:
	rm -rf results/*.json results/*.csv results/*.npz results/*.npy results/*.log results/figures results_k*/ results_ours*/
