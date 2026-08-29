import pytest

from oct_cds.data.manifest import (
    assert_no_leakage,
    build_oct_c8_manifest,
    build_optopol_manifest,
)


def test_oct_c8_manifest_shape(fake_oct_c8):
    m = build_oct_c8_manifest(fake_oct_c8, probe_images=False)
    assert set(m) == {"train", "val", "test"}
    for split, df in m.items():
        assert len(df) == 16                      # 8 classes x 2 imgs
        assert set(df["label_id"]) == set(range(8))
        assert (df["split"] == split).all()
        assert (df["dataset"] == "oct_c8").all()


def test_optopol_is_external_only(fake_optopol):
    df = build_optopol_manifest(fake_optopol, probe_images=False)
    assert (df["split"] == "external_test").all()
    assert (df["dataset"] == "clinic_optopol").all()
    assert len(df) == 5


def test_optopol_config_must_forbid_training(fake_optopol):
    bad = {**fake_optopol, "train_forbidden": False}
    with pytest.raises(ValueError):
        build_optopol_manifest(bad, probe_images=False)


def test_no_patient_leakage_across_splits(fake_oct_c8, fake_optopol):
    m = build_oct_c8_manifest(fake_oct_c8, probe_images=False)
    m["external_test"] = build_optopol_manifest(fake_optopol, probe_images=False)
    assert_no_leakage(m)                           # must not raise


def test_leakage_detector_trips_on_shared_patient(fake_oct_c8):
    m = build_oct_c8_manifest(fake_oct_c8, probe_images=False)
    # forge a shared patient_id between train and test
    m["test"].loc[0, "patient_id"] = m["train"].loc[0, "patient_id"]
    with pytest.raises(AssertionError):
        assert_no_leakage(m)
