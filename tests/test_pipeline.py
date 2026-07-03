"""Smoke + contract tests. Run: .venv\\Scripts\\python -m pytest -q

These use sample mode only (no GEE/network), so they run anywhere and guard the
numeric contracts the dashboard and README depend on.
"""

import numpy as np
import pytest

from agripulse import config
from agripulse.data_sample import generate_scene
from agripulse.features import build_features
from agripulse.classify import classify_crops
from agripulse.stress import stress_assessment
from agripulse.water import water_balance


@pytest.fixture(scope="module")
def scene():
    return generate_scene()


def test_scene_contract(scene):
    T, H, W = scene["ndvi"].shape
    # every band the pipeline consumes is present and shaped (T,H,W)
    for k in ("ndvi", "ndwi", "evi", "vv", "vh", "smi"):
        assert scene[k].shape == (T, H, W), k
        assert not np.isnan(scene[k]).any(), f"{k} has NaN after generation"


def test_sample_and_gee_scene_keys_parity():
    # GEE provider must fill the same contract as sample (minus truth-only keys)
    import agripulse.data_gee as g
    sample_keys = set(generate_scene().keys())
    for k in ("ndvi", "ndwi", "evi", "vv", "vh", "smi", "et0", "rain", "gt"):
        assert k in sample_keys, k


def test_vci_bounds_and_stress_range(scene):
    # give the scene a VCI baseline so the VCI path is exercised
    ndvi = scene["ndvi"]
    scene = {**scene, "ndvi_lo": ndvi.min(0), "ndvi_hi": ndvi.max(0)}
    features = build_features(scene)
    crop_map, conf, metrics = classify_crops(features, scene)
    stress, stage, score, vci = stress_assessment(scene, crop_map, t_now=ndvi.shape[0] - 1)
    assert vci.min() >= 0.0 and vci.max() <= 1.0
    assert set(np.unique(stress)).issubset({0, 1, 2, 3})
    assert set(np.unique(stage)).issubset({0, 1, 2, 3})
    assert 0.0 <= conf.min() and conf.max() <= 1.0


def test_metrics_are_well_formed(scene):
    features = build_features(scene)
    _, _, m = classify_crops(features, scene)
    assert 0 <= m["overall_accuracy"] <= 1
    assert -1 <= m["kappa"] <= 1
    # OA must be reported next to its no-information baseline (honesty contract)
    assert "no_information_rate" in m and "per_class_precision" in m
    assert set(m["per_class_recall"]) == set(m["per_class_precision"])


def test_advisory_classes_valid(scene):
    features = build_features(scene)
    crop_map, _, _ = classify_crops(features, scene)
    stress, stage, score, vci = stress_assessment(scene, crop_map)
    deficit, advisory, _ = water_balance(scene, crop_map, stage, stress, vci)
    assert set(np.unique(advisory)).issubset(set(config.ADVISORY_CLASSES))
    assert (deficit >= 0).all()
    # non-crop pixels are never advised to irrigate
    assert (advisory[crop_map == 0] == 0).all()


def test_stage_aware_thresholds_differ():
    # flowering must be flagged at a higher VCI than maturity (stage-awareness)
    assert config.STRESS_VCI_BANDS[2][0] > config.STRESS_VCI_BANDS[3][0]


def test_feature_names_align_with_matrix(scene):
    # importance labels are meaningless if names and columns drift apart
    from agripulse.features import feature_names
    feats = build_features(scene)
    assert len(feature_names(scene)) == feats.shape[1]


def test_per_class_metrics_are_fold_averaged(scene):
    features = build_features(scene)
    _, _, m = classify_crops(features, scene)
    # per-class F1 must carry a std (same rigor as the aggregate), keyed like the mean
    assert set(m["per_class_f1"]) == set(m["per_class_f1_std"])
    assert m["n_folds"] >= 2  # variance is meaningful only with multiple folds


def test_advisory_escalates_and_deescalates():
    # unit-test the advisory logic directly on constructed inputs
    crop = np.array([[1, 1, 1]], dtype=np.int32)
    stage = np.array([[2, 2, 2]], dtype=np.int32)      # flowering
    stress = np.array([[0, 3, 0]], dtype=np.int32)     # middle pixel severely stressed
    vci = np.array([[0.9, 0.2, 0.3]], dtype=np.float32)  # first pixel wet
    # et0=18, Kc(wheat,flowering)=1.15 -> ETc~20.7mm: a moderate deficit, above the
    # 8mm trigger but below the 35mm critical, so de-escalation can apply
    scn = {"ndvi": np.zeros((1, 1, 3)), "et0": np.array([18.0]), "rain": np.array([0.0])}
    deficit, advisory, _ = water_balance(scn, crop, stage, stress, vci, t_now=0)
    assert advisory[0, 1] == 2                          # stressed + deficit -> irrigate now
    assert advisory[0, 0] == 0                          # high VCI (wet) -> de-escalated
    assert advisory[0, 2] == 1                          # deficit, not stressed -> schedule


def test_end_to_end_sample_run(tmp_path, monkeypatch):
    # full orchestration in sample mode writes a well-formed summary + maps
    import json
    import agripulse.pipeline as pipe
    monkeypatch.setattr(pipe, "OUT", tmp_path)
    summary = pipe.run("sample")
    for key in ("metrics", "stress_by_stage_crop", "crop_area_ha",
                "recommended_depth_mm", "advisory_pct", "stress_validation"):
        assert key in summary, key
    m = summary["metrics"]
    assert "kappa_std" in m and m["n_folds"] >= 1        # variance reported
    assert "feature_importance_groups" in m              # importances reported
    saved = {p.name for p in tmp_path.glob("*.png")}
    assert {"crop_map.png", "stress_map.png", "advisory_map.png",
            "feature_importance.png"} <= saved
    # every map is georeferenced
    assert (tmp_path / "crop_map.wld").exists() and (tmp_path / "crop_map.prj").exists()
    json.loads((tmp_path / "summary.json").read_text())
