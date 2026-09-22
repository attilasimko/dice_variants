import json

import numpy as np
import SimpleITK as sitk

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
    src = tmp_path / "ACDC"
    for split, first, n in (("training", 1, 10), ("testing", 101, 2)):
        for i in range(first, first + n):
            patient = src / split / f"patient{i:03d}"
            group = "DCM" if i % 2 else "NOR"
            patient.mkdir(parents=True)
            (patient / "Info.cfg").write_text(f"ED: 1\nES: 7\nGroup: {group}\n")
            write_nifti(patient / f"patient{i:03d}_4d.nii.gz", np.zeros((8, 4, 6, 6)))
            for frame in ("01", "07"):
                name = f"patient{i:03d}_frame{frame}"
                write_nifti(patient / f"{name}.nii.gz", np.random.rand(4, 6, 6))
                write_nifti(patient / f"{name}_gt.nii.gz", np.ones((4, 6, 6)), np.uint8)

    convert.convert_acdc(src)

    out = tmp_path / "raw" / "Dataset027_ACDC"
    assert len(list((out / "imagesTr").iterdir())) == 20
    assert (out / "imagesTr" / "patient001_frame07_0000.nii.gz").exists()
    assert (out / "labelsTr" / "patient001_frame07.nii.gz").exists()
    assert len(list((out / "imagesTs").iterdir())) == 4
    assert json.loads((out / "dataset.json").read_text())["numTraining"] == 20
    splits = load_splits(tmp_path, "Dataset027_ACDC")
    assert len(splits) == convert.N_FOLDS
    assert all(len(s["val"]) == 4 for s in splits)


def test_convert_wmh(tmp_path, monkeypatch):
    set_nnunet_dirs(tmp_path, monkeypatch)
    src = tmp_path / "WMH"
    subjects = [
        *(f"training/Amsterdam/GE3T/{i}" for i in range(100, 105)),
        *(f"training/Singapore/{i}" for i in range(50, 55)),
        "test/Utrecht/7",
    ]
    for rel in subjects:
        write_nifti(src / rel / "pre" / "FLAIR.nii.gz", np.random.rand(4, 6, 6))
        write_nifti(src / rel / "pre" / "T1.nii.gz", np.random.rand(4, 6, 6))
        write_nifti(src / rel / "wmh.nii.gz", np.random.randint(0, 3, (4, 6, 6)))

    convert.convert_wmh(src)

    out = tmp_path / "raw" / "Dataset028_WMH"
    assert (out / "imagesTr" / "Amsterdam_GE3T_100_0000.nii.gz").exists()
    assert (out / "imagesTr" / "Singapore_50_0001.nii.gz").exists()
    assert (out / "imagesTs" / "Utrecht_7_0000.nii.gz").exists()
    label = sitk.ReadImage(str(out / "labelsTr" / "Singapore_50.nii.gz"))
    assert label.GetPixelID() == sitk.sitkUInt8
    dataset = json.loads((out / "dataset.json").read_text())
    assert dataset["labels"] == {"background": 0, "WMH": 1, "ignore": 2}
    assert dataset["numTraining"] == 10
    for s in load_splits(tmp_path, "Dataset028_WMH"):
        assert sorted(c.split("_")[0] for c in s["val"]) == ["Amsterdam", "Singapore"]
