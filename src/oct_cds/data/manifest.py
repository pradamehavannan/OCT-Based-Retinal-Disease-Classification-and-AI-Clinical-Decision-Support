"""Build the CSV manifests that every downstream stage consumes.

See ``data/metadata/data_dictionary.md`` for the column contract.

Two dataset families are supported:

* ``oct_c8``         - ImageFolder layout, ``root/<split>/<class_dir>/*.png``.
                       The authors' split is used verbatim (train/val/test).
* ``clinic_optopol`` - flat or single-level dir of files named per the OPTOPOL
                       convention (see ``oct_cds.data.optopol``). Always emitted
                       with ``split == external_test``; asserted, build fails
                       otherwise.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from oct_cds.common.logging import get_logger
from oct_cds.common.paths import rel_to_root
from oct_cds.data.label_map import LabelMap, load_label_map
from oct_cds.data.optopol import parse_optopol_filename

log = get_logger(__name__)

MANIFEST_COLUMNS = [
    "image_path", "label_key", "label_id", "split", "dataset", "source",
    "patient_id", "eye", "quality_flag", "width", "height", "md5",
    "near_dup_of", "grader", "grading_date", "reference_standard", "notes",
]

_IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _md5(path: Path, probe: bool) -> str:
    if not probe:
        return ""
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()


def _size(path: Path, probe: bool) -> tuple[int | None, int | None]:
    if not probe:
        return None, None
    try:
        from PIL import Image

        with Image.open(path) as im:
            return im.width, im.height
    except Exception:  # noqa: BLE001 - a broken image should not kill the build
        return None, None


# ---------------------------------------------------------------------------
# OCT-C8
# ---------------------------------------------------------------------------
def build_oct_c8_manifest(
    cfg: dict[str, Any],
    label_map: LabelMap | None = None,
    *,
    probe_images: bool = True,
) -> dict[str, pd.DataFrame]:
    lm = label_map or load_label_map(cfg.get("label_map"))
    root = Path(cfg["root"])
    class_dir_map: dict[str, str] = cfg["class_dir_map"]  # canonical_key -> dir name
    out: dict[str, pd.DataFrame] = {}

    for split, split_dir in cfg["split_dirs"].items():
        rows: list[dict[str, Any]] = []
        split_root = root / split_dir
        for canonical_key, dir_name in class_dir_map.items():
            lm.id(canonical_key)  # fail fast on a bad canonical key in the config
            cls_dir = split_root / dir_name
            if not cls_dir.is_dir() and (split_root / canonical_key).is_dir():
                cls_dir = split_root / canonical_key  # tolerate already-canonical dirs
            if not cls_dir.is_dir():
                log.warning("missing class dir: %s", cls_dir)
                continue
            for img in sorted(cls_dir.iterdir()):
                if img.suffix.lower() not in _IMG_EXT:
                    continue
                w, h = _size(img, probe_images)
                rows.append(
                    {
                        "image_path": rel_to_root(img),
                        "label_key": canonical_key,
                        "label_id": lm.id(canonical_key),
                        "split": split,
                        "dataset": "oct_c8",
                        "source": "oct_c8_mixed",
                        "patient_id": img.stem,          # per-image; no patient IDs
                        "eye": "unknown",
                        "quality_flag": "ok",
                        "width": w,
                        "height": h,
                        "md5": _md5(img, probe_images),
                        "near_dup_of": "",
                        "grader": "oct_c8_authors",
                        "grading_date": "",
                        "reference_standard": "oct_c8_label",
                        "notes": "",
                    }
                )
        df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
        out[split] = df
        log.info("oct_c8 %s: %d images", split, len(df))
    return out


# ---------------------------------------------------------------------------
# OPTOPOL clinic external set
# ---------------------------------------------------------------------------
def build_optopol_manifest(
    cfg: dict[str, Any],
    label_map: LabelMap | None = None,
    *,
    probe_images: bool = True,
) -> pd.DataFrame:
    if cfg.get("train_forbidden") is not True or "external_test" not in cfg.get("use", []):
        raise ValueError("clinic_optopol config must be external_test-only")

    lm = label_map or load_label_map(cfg.get("label_map"))
    root = Path(cfg["root"])
    rows: list[dict[str, Any]] = []

    # accept files directly under root or one level of class subdirs
    for img in sorted(root.rglob("*")):
        if img.suffix.lower() not in _IMG_EXT or not img.is_file():
            continue
        dir_label = img.parent.name if img.parent != root else None
        parsed = parse_optopol_filename(img.name, lm, dir_label=dir_label)
        w, h = _size(img, probe_images)
        rows.append(
            {
                "image_path": rel_to_root(img),
                "label_key": parsed.label_key,
                "label_id": parsed.label_id,
                "split": "external_test",
                "dataset": "clinic_optopol",
                "source": "optopol_revo",
                "patient_id": parsed.patient_id,
                "eye": parsed.eye,
                "quality_flag": "ok",
                "width": w,
                "height": h,
                "md5": _md5(img, probe_images),
                "near_dup_of": "",
                "grader": "clinic_ophthalmologist",
                "grading_date": "",
                "reference_standard": "clinical_diagnosis",
                "notes": "external validation only; de-identification reviewed",
            }
        )

    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    assert (df["split"] == "external_test").all(), "OPTOPOL rows must be external_test"
    assert (df["dataset"] == "clinic_optopol").all()
    log.info("clinic_optopol external_test: %d images", len(df))
    return df


# ---------------------------------------------------------------------------
# leakage / integrity checks
# ---------------------------------------------------------------------------
def assert_no_leakage(manifests: dict[str, pd.DataFrame]) -> None:
    """Invariants from data/metadata/data_dictionary.md."""
    internal = {s: m for s, m in manifests.items() if s in {"train", "val", "test"}}
    seen: dict[str, str] = {}
    for split, m in internal.items():
        for pid in m["patient_id"]:
            if pid in seen and seen[pid] != split:
                raise AssertionError(
                    f"patient_id {pid!r} in both {seen[pid]!r} and {split!r}"
                )
            seen[pid] = split

    if "external_test" in manifests:
        ext = manifests["external_test"]
        assert (ext["dataset"] == "clinic_optopol").all()
        assert (ext["split"] == "external_test").all()
        internal_paths = set()
        for m in internal.values():
            internal_paths |= set(m["image_path"])
        assert internal_paths.isdisjoint(set(ext["image_path"])), "external set leaks into train/val/test"


def write_manifests(manifests: dict[str, pd.DataFrame], cfg: dict[str, Any]) -> None:
    targets: dict[str, str] = cfg["manifest"]
    for split, df in manifests.items():
        if split not in targets:
            continue
        dest = Path(targets[split])
        dest.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(dest, index=False)
        log.info("wrote %s (%d rows)", dest, len(df))
