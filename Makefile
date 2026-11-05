# Pipeline entry points. Run `make help` for a list.
#
# BACKEND=ours (the default) uses the from-scratch NumPy code in src/nsch_subtypes/
# and writes to results/. BACKEND=reference uses StepMix / SciPy / scikit-learn
# and writes to results_reference/, as the check on our code.

PY      ?= uv run python
BACKEND ?= ours
# Use the same backend-specific result folders as sweep_k.sh.
FIT_RESULTS = $(if $(filter reference,$(BACKEND)),results_reference,results)
STARTS  ?= 20
NBOOT   ?= 200
KS      ?= 4 5 6 7 8
K       ?= 4

.PHONY: help all download fit sweep choose calibrate outcomes figures compare overview paper test smoke lint clean

help:
	@echo "make test       unit tests on synthetic data (seconds)"
	@echo "make smoke      run every pipeline step once at tiny settings (about 5 minutes)"
	@echo "make fit        step 3: latent class models K=1..8 into results/"
	@echo "make sweep      steps 4 to 15 at K in KS ($(KS)) and the figures (hours; mostly bootstrap)"
	@echo "make choose     step 13 only: apply the K rule to the bootstraps on file"
	@echo "make calibrate  steps 14 and 15 only: shape, stability and replication checks on simulated data (about an hour; step 15 at K=$(K))"
	@echo "make outcomes   step 16 only: the three held-out outcomes and the pandemic-years checks at K=$(K) (5 to 10 minutes)"
	@echo "make figures    redraw the results figures from existing outputs"
	@echo "make compare    compare saved reference and from-scratch fits, with tables, data, and diagrams"
	@echo "make overview   the data-overview figures (needs data/interim from 'make download')"
	@echo "make paper      build paper/main.pdf from the committed figures (needs LaTeX)"
	@echo "make download   steps 1 and 2 from the Census files (about 60 MB; not needed for anything else)"
	@echo "make all        fit + sweep"
	@echo "Options: BACKEND=$(BACKEND) STARTS=$(STARTS) NBOOT=$(NBOOT) KS='$(KS)' K=$(K)"

all: fit sweep

# data/processed/analysis.parquet and codebook.csv are committed, so nothing
# below needs the download. Rebuild them on purpose with `make download`.
download:
	$(PY) scripts/01_select.py --download
	$(PY) scripts/02_encode.py

fit:
	$(PY) scripts/03_fit_lca.py --backend $(BACKEND) --results $(FIT_RESULTS) --n-starts $(STARTS) --permutation-null

sweep:
	BACKEND=$(BACKEND) STARTS=$(STARTS) NBOOT=$(NBOOT) bash scripts/sweep_k.sh $(KS)

choose:
	$(PY) scripts/13_heldout_selection.py --backend $(BACKEND) --results $(FIT_RESULTS) --reuse-heldout

calibrate:
	$(PY) scripts/14_shape_and_stability.py --backend $(BACKEND) --results $(FIT_RESULTS)
	$(PY) scripts/15_replication_calibration.py --backend $(BACKEND) --results $(FIT_RESULTS)_k$(K) --k $(K)

outcomes:
	$(PY) scripts/16_outcomes.py --backend $(BACKEND) --results $(FIT_RESULTS)_k$(K) --k $(K)

figures:
	$(PY) scripts/make_results_figures.py --results $(FIT_RESULTS)

compare:
	$(PY) scripts/compare_backends.py

overview:
	$(PY) scripts/make_data_overview.py

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
	@n=$$(grep -l 'begin{stub}' paper/sections/*.tex | wc -l); echo "sections with text still to write: $$n"

test:
	uv run pytest -q -m "not slow"

smoke:
	uv run pytest -q -m slow

lint:
	uv run ruff check . && uv run ruff format --check .

clean:
	rm -rf results/*.json results/*.csv results/*.npz results/*.npy results/*.log results/figures results_k*/ results_reference*/
