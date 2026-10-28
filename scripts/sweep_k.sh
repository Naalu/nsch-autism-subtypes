#!/usr/bin/env bash
# Run every check at several K, then choose K and draw the figures.
#
# Needs: uv on PATH, and the step 3 fits in the main results folder
# (results/lca_k1.npz ... lca_k8.npz). If they are missing, step 3 is run
# first at full settings (K = 1 to 8, 20 starts, shuffled baseline; about
# 5 minutes). Everything else reads the committed data/processed files, so
# no download is needed.
#
# For each K listed this creates results_k{K}/ holding a copy of that K's
# step 3 fit and runs steps 4 to 12 against it (the 200-resample bootstrap,
# step 6, last because it is slow: roughly 20 to 40 minutes per K). Then
# step 13 applies the K rule across every folder that has a bootstrap,
# step 14 calibrates the shape and stability checks by simulation (about an
# hour), and the results figures are drawn for the chosen K.
#
# With BACKEND=ours the folders are results_ours/ and results_ours_k{K}/, and
# the summary also prints how closely the from-scratch fits match the
# reference ones.
#
# Usage, from the repository root:
#     bash scripts/sweep_k.sh                # K = 4 5 6 7, reference backend
#     bash scripts/sweep_k.sh 4 5            # your own list
#     BACKEND=ours bash scripts/sweep_k.sh 4 5
#
# Progress and every step's output go to <main results folder>/sweep.log.
# Each K folder gets a STATUS file: "complete", or the list of steps that failed.

set -uo pipefail
cd "$(dirname "$0")/.."

KS=("$@")
if [ ${#KS[@]} -eq 0 ]; then KS=(4 5 6 7); fi
BACKEND="${BACKEND:-reference}"
STARTS="${STARTS:-20}"
NBOOT="${NBOOT:-200}"
PY="uv run python"

if [ "$BACKEND" = "ours" ]; then MAIN="results_ours"; PREFIX="results_ours_k"; else MAIN="results"; PREFIX="results_k"; fi
mkdir -p "$MAIN/figures"
LOG="$MAIN/sweep.log"

# Keep a Mac awake for the duration (no effect elsewhere).
if command -v caffeinate >/dev/null 2>&1; then caffeinate -i -w $$ & fi

FAILED=()
step() {  # step <name> <command...>: run, log, record failures, keep going
    local name="$1"; shift
    echo "=== $(date '+%F %T')  $name" | tee -a "$LOG"
    if "$@" >>"$LOG" 2>&1; then
        echo "    done" | tee -a "$LOG"
    else
        echo "    FAILED (see $LOG)" | tee -a "$LOG"
        FAILED+=("$name")
    fi
}

echo "sweep started $(date '+%F %T'); K = ${KS[*]}; backend = $BACKEND" | tee -a "$LOG"

# 0. Step 3, if this backend has no fits yet. Nothing else can run without it.
if [ ! -f "$MAIN/lca_fits.json" ]; then
    step "step 3, latent class fits K=1..8" $PY scripts/03_fit_lca.py --results "$MAIN" --backend "$BACKEND" --n-starts "$STARTS" --permutation-null
    if [ ! -f "$MAIN/lca_fits.json" ]; then echo "step 3 did not finish; stopping" | tee -a "$LOG"; exit 1; fi
fi

# 1. Every check at each requested K.
for K in "${KS[@]}"; do
    DIR="$PREFIX$K"
    FAILED=()
    mkdir -p "$DIR/figures"
    if ! cp "$MAIN/lca_k$K.npz" "$DIR/"; then
        echo "no step 3 fit for K=$K in $MAIN; skipping" | tee -a "$LOG"
        echo "skipped: no step 3 fit" > "$DIR/STATUS"
        continue
    fi
    common=(--results "$DIR" --backend "$BACKEND" --k "$K")
    step "K=$K second method, k-means"     $PY scripts/04_crosscheck.py "${common[@]}" --method kmeans
    step "K=$K second method, hierarchical" $PY scripts/04_crosscheck.py "${common[@]}" --method hierarchical
    step "K=$K other methods"              $PY scripts/04_crosscheck.py "${common[@]}" --method alternatives
    step "K=$K held-out years"             $PY scripts/05_replicate.py "${common[@]}" --n-starts "$STARTS"
    step "K=$K describe and severity test" $PY scripts/07_describe.py --results "$DIR" --k "$K"
    step "K=$K 2025 forward test"          $PY scripts/08_forward_2025.py "${common[@]}" --n-starts "$STARTS"
    step "K=$K age bands"                  $PY scripts/09_age_bands.py "${common[@]}" --n-starts "$STARTS"
    step "K=$K 'ever told' items"          $PY scripts/10_parallel_ever.py "${common[@]}" --n-starts "$STARTS"
    step "K=$K robustness"                 $PY scripts/11_robustness.py "${common[@]}" --n-starts "$STARTS"
    step "K=$K classes vs dimensions"      $PY scripts/12_dimensions.py "${common[@]}" --fits "$MAIN"
    step "K=$K bootstrap ($NBOOT)"         $PY scripts/06_bootstrap.py "${common[@]}" --n-boot "$NBOOT" --n-starts "$STARTS"
    if [ ${#FAILED[@]} -eq 0 ]; then echo "complete" > "$DIR/STATUS"; else printf "failed: %s\n" "${FAILED[@]}" > "$DIR/STATUS"; fi
done

# 2. Choose K (held-out likelihood is computed once and reused), calibrate by simulation, then draw.
if [ -f "$MAIN/heldout_selection.csv" ]; then REUSE=(--reuse-heldout); else REUSE=(); fi
step "step 13, held-out likelihood and the K rule" $PY scripts/13_heldout_selection.py --results "$MAIN" --backend "$BACKEND" "${REUSE[@]}"
step "step 14, shape of the latent space and stability under known truth" $PY scripts/14_shape_and_stability.py --results "$MAIN" --backend "$BACKEND"
if [ "$BACKEND" = "ours" ]; then REPORTS=(--reports "$MAIN/reports"); else REPORTS=(); fi
step "results figures" $PY scripts/make_results_figures.py --results "$MAIN" "${REPORTS[@]}"

# 3. Summary.
echo "=== summary" | tee -a "$LOG"
cat reports/tables/checks_by_k.csv 2>/dev/null | tee -a "$LOG"
for K in "${KS[@]}"; do echo "$PREFIX$K: $(cat "$PREFIX$K/STATUS" 2>/dev/null)" | tee -a "$LOG"; done
if [ "$BACKEND" = "ours" ]; then
    PREFIX="$PREFIX" KS="${KS[*]}" $PY - <<'EOF' | tee -a "$LOG"
import os
import numpy as np
from nsch_subtypes.validate import adjusted_rand_index, match_classes, relabel_to_match

print("agreement between the from-scratch and reference fits:")
for k in os.environ["KS"].split():
    try:
        a, b = np.load(f"results_k{k}/lca_k{k}.npz"), np.load(f"{os.environ['PREFIX']}{k}/lca_k{k}.npz")
    except FileNotFoundError:
        print(f"  K={k}: one of the two runs is missing")
        continue
    perm = match_classes(a["condition_rates"], b["condition_rates"])
    ari = adjusted_rand_index(a["responsibilities"].argmax(1), relabel_to_match(b["responsibilities"].argmax(1), perm))
    gap = np.abs(a["condition_rates"] - b["condition_rates"][perm]).max()
    print(f"  K={k}: ARI {ari:.3f}, largest rate difference {gap:.3f}")
EOF
fi
echo "sweep finished $(date '+%F %T')" | tee -a "$LOG"
