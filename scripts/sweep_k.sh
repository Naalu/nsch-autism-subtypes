#!/usr/bin/env bash
# Run every check at several K without touching the main results/ folder.
#
# For each K listed, copies the step 3 fits into results_k{K}/ and runs
# steps 4 (both methods), 5, 7, 8, 9, 10, 11, 12 and finally 6 (the slow
# bootstrap) against that K. Also adds the k-means cross-check to the main
# folder at the K BIC chose.
#
# With BACKEND=ours everything goes to results_ours/ and results_ours_k{K}/
# instead (running step 3 first if needed), and the summary prints the
# agreement between the two backends at each K.
#
# Usage, from the repository root:
#     bash scripts/sweep_k.sh                # K = 4 5 6, reference backend
#     bash scripts/sweep_k.sh 4 5            # your own list
#     BACKEND=ours bash scripts/sweep_k.sh 4 5
#
# Output of every step is appended to <main results dir>/sweep.log.

set -euo pipefail
cd "$(dirname "$0")/.."

KS=("$@")
if [ ${#KS[@]} -eq 0 ]; then KS=(4 5 6); fi
BACKEND="${BACKEND:-reference}"
STARTS="${STARTS:-20}"
NBOOT="${NBOOT:-200}"
PY="uv run python"

if [ "$BACKEND" = "ours" ]; then MAIN="results_ours"; PREFIX="results_ours_k"; else MAIN="results"; PREFIX="results_k"; fi
mkdir -p "$MAIN/figures"
LOG="$MAIN/sweep.log"

# Keep the Mac awake for the duration if caffeinate exists (macOS only).
if command -v caffeinate >/dev/null 2>&1; then
    caffeinate -i -w $$ &
fi

step() {  # step <name> <command...>: run, log, keep going if one step fails
    local name="$1"; shift
    echo "=== $(date '+%F %T')  $name" | tee -a "$LOG"
    if "$@" >>"$LOG" 2>&1; then
        echo "    done" | tee -a "$LOG"
    else
        echo "    FAILED (see $LOG)" | tee -a "$LOG"
    fi
}

echo "sweep started $(date '+%F %T'); K = ${KS[*]}; backend = $BACKEND" | tee -a "$LOG"

# 0. Step 3 if this backend has not been fit yet.
if [ ! -f "$MAIN/lca_best.json" ]; then
    step "step 3, latent class fits K=1..8" $PY scripts/03_fit_lca.py --results "$MAIN" --backend "$BACKEND" --n-starts "$STARTS" --permutation-null
fi

# 1. The cross-check at the K BIC chose.
step "k-means cross-check at BIC's K" $PY scripts/04_crosscheck.py --results "$MAIN" --backend "$BACKEND" --method kmeans --n-starts "$STARTS"

# 2. Every check at each requested K.
for K in "${KS[@]}"; do
    DIR="$PREFIX$K"
    mkdir -p "$DIR/figures"
    cp "$MAIN"/lca_k*.npz "$MAIN"/lca_best.json "$MAIN"/lca_bic.csv "$DIR/"

    step "K=$K cross-check, k-means"      $PY scripts/04_crosscheck.py   --results "$DIR" --backend "$BACKEND" --k "$K" --method kmeans --n-starts "$STARTS"
    step "K=$K cross-check, hierarchical" $PY scripts/04_crosscheck.py   --results "$DIR" --backend "$BACKEND" --k "$K" --method hierarchical
    step "K=$K replicate across years"    $PY scripts/05_replicate.py    --results "$DIR" --backend "$BACKEND" --k "$K" --n-starts "$STARTS"
    step "K=$K describe + count test"     $PY scripts/07_describe.py     --results "$DIR" --k "$K"
    step "K=$K forward test on 2025"      $PY scripts/08_forward_2025.py --results "$DIR" --backend "$BACKEND" --k "$K" --n-starts "$STARTS"
    step "K=$K age bands"                 $PY scripts/09_age_bands.py    --results "$DIR" --backend "$BACKEND" --k "$K" --n-starts "$STARTS"
    step "K=$K parallel ('ever' items)"   $PY scripts/10_parallel_ever.py --results "$DIR" --backend "$BACKEND" --k "$K" --n-starts "$STARTS"
    step "K=$K robustness"                $PY scripts/11_robustness.py   --results "$DIR" --backend "$BACKEND" --k "$K" --n-starts "$STARTS"
    step "K=$K dimensions"                $PY scripts/12_dimensions.py   --results "$DIR" --backend "$BACKEND" --k "$K"
    step "K=$K bootstrap ($NBOOT)"        $PY scripts/06_bootstrap.py    --results "$DIR" --backend "$BACKEND" --k "$K" --n-boot "$NBOOT" --n-starts "$STARTS"
done

# 3. One table to read afterwards.
echo "=== summary" | tee -a "$LOG"
PREFIX="$PREFIX" KS="${KS[*]}" $PY - <<'EOF' | tee -a "$LOG"
import json, os
import numpy as np, pandas as pd

prefix = os.environ["PREFIX"]
ks = [int(k) for k in os.environ["KS"].split()]
main = "results_ours" if "ours" in prefix else "results"

def load(d, name):
    try:
        return json.load(open(f"{d}/{name}"))
    except FileNotFoundError:
        return {}

rows = []
for d, k in [(main, None)] + [(f"{prefix}{k}", k) for k in ks]:
    km, rep, fwd, age, par, rob = (load(d, n) for n in ["crosscheck_kmeans.json", "replicate.json", "forward_2025.json", "age_bands.json", "parallel_ever.json", "robustness.json"])
    k = k or km.get("k") or rep.get("k")
    try:
        stab = pd.read_csv(f"{d}/stability.csv")
        stab_txt = f"{stab['mean_jaccard'].min():.2f} to {stab['mean_jaccard'].max():.2f}, {int((stab['mean_jaccard'] >= 0.75).sum())}/{len(stab)} stable"
    except FileNotFoundError:
        stab_txt = ""
    try:
        cvc = pd.read_csv(f"{d}/count_vs_class.csv").set_index("model")
        cvc_txt = f"{cvc.loc['count','bic']:.0f} / {cvc.loc['class','bic']:.0f} / {cvc.loc['both','bic']:.0f}"
    except FileNotFoundError:
        cvc_txt = ""
    bands = age.get("bands", {})
    rows.append({
        "dir": d, "K": k,
        "kmeans": round(km.get("ari", float("nan")), 3),
        "replicate": round(rep.get("ari", float("nan")), 3),
        "2025": round(fwd.get("ari", float("nan")), 3),
        "age 6-11": round(bands.get("6_11", {}).get("ari_pooled_vs_band_fit", float("nan")), 3),
        "age 12-17": round(bands.get("12_17", {}).get("ari_pooled_vs_band_fit", float("nan")), 3),
        "ever": round(par.get("ari_current_vs_ever", float("nan")), 3),
        "weighted": round(rob.get("weighted_fit", {}).get("ari_vs_unweighted", float("nan")), 3),
        "bootstrap": stab_txt,
        "severity BIC count/class/both": cvc_txt,
    })
pd.set_option("display.width", 250)
print(pd.DataFrame(rows).to_string(index=False))
print("pass line for every ARI: 0.65; stable class: mean Jaccard >= 0.75")

# Backend agreement, when both have been run.
if "ours" in prefix:
    from nsch_subtypes.validate import adjusted_rand_index, match_classes, relabel_to_match
    print("\nagreement between the from-scratch and reference fits:")
    for k in ks:
        try:
            a = np.load(f"results_k{k}/lca_k{k}.npz"); b = np.load(f"results_ours_k{k}/lca_k{k}.npz")
        except FileNotFoundError:
            print(f"  K={k}: reference run not found"); continue
        perm = match_classes(a["condition_rates"], b["condition_rates"])
        ari = adjusted_rand_index(a["responsibilities"].argmax(1), relabel_to_match(b["responsibilities"].argmax(1), perm))
        gap = np.abs(a["condition_rates"] - b["condition_rates"][perm]).max()
        print(f"  K={k}: ARI {ari:.3f}, largest rate difference {gap:.3f}")
EOF

echo "sweep finished $(date '+%F %T')" | tee -a "$LOG"
