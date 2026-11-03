"""Pin the K=4 fit on the committed analysis table.

A silent change to the EM (a different clip, start distribution or stopping
rule) would move these numbers without failing any synthetic test. The
expected values are the ones in ``reports/tables/backend_comparison.csv``.
About six seconds, so it runs with the slow tests.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from _common import ANALYSIS_PATH, MAIN_YEARS, load_analysis  # noqa: E402
from nsch_subtypes.data import to_flag_matrix  # noqa: E402
from nsch_subtypes.lca import fit_lca  # noqa: E402

pytestmark = pytest.mark.slow


@pytest.mark.skipif(not ANALYSIS_PATH.exists(), reason="analysis table not built")
def test_k4_fit_on_the_real_data_is_unchanged():
    table, flags = load_analysis(MAIN_YEARS)
    X = to_flag_matrix(table, flags)
    assert X.shape == (8891, 18)
    fit = fit_lca(X, 4, n_starts=20, seed=0)
    assert fit.converged
    assert fit.log_likelihood == pytest.approx(-52872.7535, abs=0.01)
    assert np.sort(fit.class_sizes) == pytest.approx([0.1676, 0.2183, 0.2898, 0.3243], abs=0.001)
