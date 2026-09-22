"""Raw ACDC / WMH data -> nnU-Net raw datasets and patient-level splits_final.json.

ACDC: <acdc>/{training,testing}/patientXXX/patientXXX_frameYY[_gt].nii.gz + Info.cfg.
      One case per annotated frame (ED, ES); both frames of a patient share a fold,
      and folds are balanced over the five pathology groups.
WMH:  every folder under <wmh> holding wmh.nii.gz is a subject with pre/FLAIR.nii.gz
      and pre/T1.nii.gz (MICCAI 2017 layout). Subjects below a "test" folder go to
      imagesTs/labelsTs. Folds are balanced over sites. Label 2 ("other pathology",
      not scored in the challenge) becomes nnU-Net's ignore label.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

DATASETS = {"ACDC": "Dataset027_ACDC", "WMH": "Dataset028_WMH"}
N_FOLDS = 5
SPLIT_SEED = 1234


def kfold_splits(
    strata: dict[str, str], cases: dict[str, list[str]]
) -> list[dict[str, list[str]]]:
    """Folds over patients, balanced within each stratum.

    strata maps patient -> stratum, cases maps patient -> its case identifiers.
    """
    rng = np.random.default_rng(SPLIT_SEED)
    fold = {}
    for stratum in sorted(set(strata.values())):
        patients = sorted(p for p, s in strata.items() if s == stratum)
        for i, j in enumerate(rng.permutation(len(patients))):
            fold[patients[j]] = i % N_FOLDS
    return [
        {
            "train": sorted(c for p in cases if fold[p] != f for c in cases[p]),
            "val": sorted(c for p in cases if fold[p] == f for c in cases[p]),
        }
        for f in range(N_FOLDS)
    ]


def make_dataset_dir(name: str) -> Path:
    out = Path(os.environ["nnUNet_raw"]) / name
    shutil.rmtree(out, ignore_errors=True)
    for sub in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        (out / sub).mkdir(parents=True)
    return out


def write_splits(name: str, splits: list[dict[str, list[str]]]) -> None:
    path = Path(os.environ["nnUNet_preprocessed"]) / name / "splits_final.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=2))


def convert_acdc(src: Path) -> None:
    out = make_dataset_dir(DATASETS["ACDC"])
    strata, cases = {}, {}
    for split, suffix in (("training", "Tr"), ("testing", "Ts")):
        for patient in sorted(p for p in (src / split).iterdir() if p.is_dir()):
            frames = [
                gt.name.removesuffix("_gt.nii.gz")
                for gt in sorted(patient.glob("*_frame??_gt.nii.gz"))
            ]
            for frame in frames:
                shutil.copy(
                    patient / f"{frame}.nii.gz",
                    out / f"images{suffix}" / f"{frame}_0000.nii.gz",
                )
                shutil.copy(
                    patient / f"{frame}_gt.nii.gz",
                    out / f"labels{suffix}" / f"{frame}.nii.gz",
                )
            if split == "training":
                info = dict(
                    line.split(": ", 1)
                    for line in (patient / "Info.cfg").read_text().splitlines()
                    if ": " in line
                )
                strata[patient.name] = info["Group"].strip()
                cases[patient.name] = frames

    generate_dataset_json(
        str(out),
        channel_names={0: "cineMRI"},
        labels={"background": 0, "RV": 1, "MLV": 2, "LVC": 3},
        num_training_cases=sum(len(c) for c in cases.values()),
        file_ending=".nii.gz",
        dataset_name=DATASETS["ACDC"],
    )
    write_splits(DATASETS["ACDC"], kfold_splits(strata, cases))


def convert_wmh(src: Path) -> None:
    out = make_dataset_dir(DATASETS["WMH"])
    strata = {}
    for subject in sorted(p.parent for p in src.rglob("wmh.nii.gz")):
        rel = subject.relative_to(src).parts
        is_test = any(p.lower() == "test" for p in rel)
        # e.g. training/Amsterdam/GE3T/100 -> case Amsterdam_GE3T_100, site Amsterdam_GE3T
        parts = [p for p in rel if p.lower() not in ("training", "test")]
        case = "_".join(parts)
        suffix = "Ts" if is_test else "Tr"
        shutil.copy(
            subject / "pre" / "FLAIR.nii.gz",
            out / f"images{suffix}" / f"{case}_0000.nii.gz",
        )
        shutil.copy(
            subject / "pre" / "T1.nii.gz",
            out / f"images{suffix}" / f"{case}_0001.nii.gz",
        )
        label = sitk.Cast(sitk.ReadImage(str(subject / "wmh.nii.gz")), sitk.sitkUInt8)
        sitk.WriteImage(label, str(out / f"labels{suffix}" / f"{case}.nii.gz"))
        if not is_test:
            strata[case] = "_".join(parts[:-1])

    generate_dataset_json(
        str(out),
        channel_names={0: "FLAIR", 1: "T1"},
        labels={"background": 0, "WMH": 1, "ignore": 2},
        num_training_cases=len(strata),
        file_ending=".nii.gz",
        dataset_name=DATASETS["WMH"],
    )
    write_splits(DATASETS["WMH"], kfold_splits(strata, {c: [c] for c in strata}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--acdc", type=Path, help="ACDC folder with training/, testing/"
    )
    parser.add_argument("--wmh", type=Path, help="WMH folder (searched recursively)")
    args = parser.parse_args()
    if args.acdc:
        convert_acdc(args.acdc)
    if args.wmh:
        convert_wmh(args.wmh)
