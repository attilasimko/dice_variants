import json
import pickle

import numpy as np
import pytest
import SimpleITK as sitk
from PIL import Image

import convert


def write_nifti(path, array, dtype=np.float32):
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(sitk.GetImageFromArray(array.astype(dtype)), str(path))


def set_nnunet_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("nnUNet_raw", str(tmp_path / "raw"))
    monkeypatch.setenv("nnUNet_preprocessed", str(tmp_path / "preprocessed"))


def load_splits(tmp_path, name):
    return json.loads(
        (tmp_path / "preprocessed" / name / "splits_final.json").read_text()
    )


def test_kfold_splits_groups_patients_and_balances_strata():
    strata = {f"p{i:02d}": ["A", "B", "C"][i % 3] for i in range(30)}
    cases = {p: [f"{p}_frame01", f"{p}_frame12"] for p in strata}

    splits = convert.kfold_splits(strata, cases)

    assert splits == convert.kfold_splits(strata, cases)
    all_cases = sorted(c for cs in cases.values() for c in cs)
    assert sorted(c for s in splits for c in s["val"]) == all_cases
    for s in splits:
        assert sorted(s["train"] + s["val"]) == all_cases
        val_patients = {c.split("_")[0] for c in s["val"]}
        assert all(f"{p}_frame12" in s["val"] for p in val_patients)
        assert sorted(strata[p] for p in val_patients) == ["A", "A", "B", "B", "C", "C"]


def test_convert_acdc(tmp_path, monkeypatch):
    set_nnunet_dirs(tmp_path, monkeypatch)
    src = tmp_path / "ACDC-2D-CL"
    rng = np.random.default_rng(0)
    spacing = {}
    for split, patients in (("train", [1, 3]), ("val", [2]), ("test", [4])):
        for sub in ("img", "gt"):
            (src / split / sub).mkdir(parents=True)
        for i in patients:
            for frame in ("01", "12"):
                for z in range(3):
                    name = f"patient{i:03d}_{frame}_0_{z:02d}"
                    img = rng.integers(0, 255, (5, 7), dtype=np.uint8)
                    img[0, 0] = z  # slice order marker
                    Image.fromarray(img).save(src / split / "img" / f"{name}.png")
                    gt = np.full((5, 7), 3, np.uint8)
                    Image.fromarray(gt).save(src / split / "gt" / f"{name}.png")
                    spacing[name] = (
                        np.float64(10.0),
                        np.float64(1.5),
                        np.float64(1.25),
                    )
    (src / "spacing_3d.pkl").write_bytes(pickle.dumps(spacing))

    convert.convert_acdc(src)

    out = tmp_path / "raw" / "Dataset027_ACDC"
    assert len(list((out / "imagesTr").iterdir())) == 6
    assert len(list((out / "labelsTs").iterdir())) == 2
    image = sitk.ReadImage(str(out / "imagesTr" / "patient003_frame12_0000.nii.gz"))
    assert image.GetSize() == (7, 5, 3)
    # PNG rows run along NIfTI x, which has the pickle's last spacing
    assert image.GetSpacing() == (1.5, 1.25, 10.0)
    assert list(sitk.GetArrayFromImage(image)[:, 0, 0]) == [0, 1, 2]
    label = sitk.ReadImage(str(out / "labelsTr" / "patient003_frame12.nii.gz"))
    assert np.all(sitk.GetArrayFromImage(label) == 3)
    assert json.loads((out / "dataset.json").read_text())["numTraining"] == 6
    assert load_splits(tmp_path, "Dataset027_ACDC") == [
        {
            "train": [
                "patient001_frame01", "patient001_frame12",
                "patient003_frame01", "patient003_frame12",
            ],
            "val": ["patient002_frame01", "patient002_frame12"],
        }
    ]  # fmt: skip


def test_convert_wmh(tmp_path, monkeypatch):
    set_nnunet_dirs(tmp_path, monkeypatch)
    src = tmp_path / "WMH"
    subjects = [
        *(f"training/Amsterdam/GE3T/{i}" for i in range(100, 105)),
        *(f"training/Singapore/{i}" for i in range(50, 55)),
        "test/Amsterdam/Philips_VU .PETMR_01./168",
    ]
    for rel in subjects:
        write_nifti(src / rel / "pre" / "FLAIR.nii.gz", np.random.rand(4, 6, 6))
        write_nifti(src / rel / "pre" / "T1.nii.gz", np.random.rand(4, 6, 6))
        write_nifti(src / rel / "wmh.nii.gz", np.random.randint(0, 3, (4, 6, 6)))
    # float noise in the header, as in the real data
    t1_path = src / "training/Singapore/50/pre/T1.nii.gz"
    t1 = sitk.ReadImage(str(t1_path))
    t1.SetDirection(np.array(t1.GetDirection()) + 2e-5)
    sitk.WriteImage(t1, str(t1_path))
    (src / "additional_annotations/observer_o3/training/Singapore/50").mkdir(
        parents=True
    )

    convert.convert_wmh(src)

    out = tmp_path / "raw" / "Dataset028_WMH"
    assert (out / "imagesTr" / "Amsterdam_GE3T_100_0000.nii.gz").exists()
    assert (out / "imagesTr" / "Singapore_50_0001.nii.gz").exists()
    assert (out / "imagesTs" / "Amsterdam_Philips_VU_PETMR_01_168_0000.nii.gz").exists()
    label = sitk.ReadImage(str(out / "labelsTr" / "Singapore_50.nii.gz"))
    assert label.GetPixelID() == sitk.sitkUInt8
    flair = sitk.ReadImage(str(out / "imagesTr" / "Singapore_50_0000.nii.gz"))
    t1 = sitk.ReadImage(str(out / "imagesTr" / "Singapore_50_0001.nii.gz"))
    assert t1.GetDirection() == flair.GetDirection() == label.GetDirection()
    dataset = json.loads((out / "dataset.json").read_text())
    assert dataset["labels"] == {"background": 0, "WMH": 1, "ignore": 2}
    assert dataset["numTraining"] == 10
    for s in load_splits(tmp_path, "Dataset028_WMH"):
        assert sorted(c.split("_")[0] for c in s["val"]) == ["Amsterdam", "Singapore"]


def test_read_aligned_rejects_real_misalignment(tmp_path):
    reference = sitk.GetImageFromArray(np.zeros((4, 6, 6), np.float32))
    shifted = sitk.GetImageFromArray(np.zeros((4, 6, 6), np.float32))
    shifted.SetOrigin((0.0, 0.0, 3.0))
    sitk.WriteImage(shifted, str(tmp_path / "t1.nii.gz"))
    with pytest.raises(ValueError, match="GetOrigin"):
        convert.read_aligned(tmp_path / "t1.nii.gz", reference)
