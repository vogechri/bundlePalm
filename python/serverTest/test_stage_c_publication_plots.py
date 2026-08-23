import json

import pytest

from build_stage_c_publication_plots import load_panels, method_class


def test_method_class_distinguishes_k1_from_k16():
    assert method_class("DRS K1 BAE-style") == "K1 diagnostic"
    assert method_class("DRS K16") == "DRS only"
    assert method_class("DRS K16 + terminal correction") == "Terminal correction"


def test_load_panels_rejects_method_coverage_drift(tmp_path):
    summary = {
        "all15_1dsfm": [],
        "all29_bal": [],
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="publication method coverage mismatch"):
        load_panels(path)
