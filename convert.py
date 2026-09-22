"""Raw ACDC-2D-CL / WMH data -> nnU-Net raw datasets and splits_final.json.

ACDC-2D-CL: {train,val,test}/{img,gt}/patientXXX_FF_0_SS.png, one 256x256 PNG per
      slice SS of frame FF (ED and ES of the 100 ACDC training patients), and
      spacing_3d.pkl mapping each slice to (z, y, x) spacing in NIfTI axes (a PNG
      row runs along NIfTI x). Slices are stacked back into one volume per frame.
      Fold 0 is the provided train/val split (by patient); test goes to imagesTs.
WMH:  every folder under <wmh> holding wmh.nii.gz is a subject with pre/FLAIR.nii.gz
      and pre/T1.nii.gz (MICCAI 2017 layout; additional_annotations/ has none).
      Subjects below a "test" folder go to imagesTs/labelsTs. 5 folds, balanced
      over sites. Label 2 ("other pathology", not scored in the challenge) becomes
      nnU-Net's ignore label.
"""

import argparse
import json
import os
import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from PIL import Image

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
    """Empty image/label folders. Only the .nii.gz files of a previous conversion are
    deleted, not the folders: on NFS, a file still open elsewhere leaves a .nfsXXXX
    placeholder that makes rmdir fail (nnU-Net ignores it)."""
    out = Path(os.environ["nnUNet_raw"]) / name
    for sub in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        (out / sub).mkdir(parents=True, exist_ok=True)
        for f in (out / sub).glob("*.nii.gz"):
            f.unlink()
    return out


def write_splits(name: str, splits: list[dict[str, list[str]]]) -> None:
    path = Path(os.environ["nnUNet_preprocessed"]) / name / "splits_final.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=2))


def stack_slices(
    pngs: list[Path], spacing_zyx: tuple[float, float, float]
) -> sitk.Image:
    image = sitk.GetImageFromArray(np.stack([np.array(Image.open(p)) for p in pngs]))
    z, y, x = map(float, spacing_zyx)
    # array axes (slice, row, col) are sitk (z, y, x); PNG rows run along NIfTI x
    image.SetSpacing((y, x, z))
    return image


def convert_acdc(src: Path) -> None:
    out = make_dataset_dir(DATASETS["ACDC"])
    spacing = pickle.loads((src / "spacing_3d.pkl").read_bytes())
    cases = {"train": [], "val": [], "test": []}
    for split, split_cases in cases.items():
        volumes = defaultdict(list)
        for png in sorted((src / split / "img").glob("*.png")):
            volumes[png.stem.rsplit("_", 1)[0]].append(png)
        for volume, pngs in volumes.items():
            patient, frame, _ = volume.split("_")
            case = f"{patient}_frame{frame}"
            suffix = "Ts" if split == "test" else "Tr"
            spacing_zyx = spacing[pngs[0].stem]
            image = stack_slices(pngs, spacing_zyx)
            label = stack_slices(
                [src / split / "gt" / p.name for p in pngs], spacing_zyx
            )
            sitk.WriteImage(image, str(out / f"images{suffix}" / f"{case}_0000.nii.gz"))
            sitk.WriteImage(label, str(out / f"labels{suffix}" / f"{case}.nii.gz"))
            split_cases.append(case)

    generate_dataset_json(
        str(out),
        channel_names={0: "cineMRI"},
        labels={"background": 0, "RV": 1, "MLV": 2, "LVC": 3},
        num_training_cases=len(cases["train"]) + len(cases["val"]),
        file_ending=".nii.gz",
        dataset_name=DATASETS["ACDC"],
    )
    write_splits(DATASETS["ACDC"], [{"train": cases["train"], "val": cases["val"]}])


def read_aligned(path: Path, reference: sitk.Image) -> sitk.Image:
    """The image at path with reference's header. WMH images and labels of a subject
    differ by float noise (up to ~2e-5 in the direction cosines), which nnU-Net
    reports as a misalignment for every case."""
    image = sitk.ReadImage(str(path))
    for get in ("GetSize", "GetSpacing", "GetOrigin", "GetDirection"):
        if not np.allclose(getattr(image, get)(), getattr(reference, get)(), atol=1e-3):
            raise ValueError(f"{path} does not align with the FLAIR ({get})")
    image.CopyInformation(reference)
    return image


def convert_wmh(src: Path) -> None:
    out = make_dataset_dir(DATASETS["WMH"])
    strata = {}
    for subject in sorted(p.parent for p in src.rglob("wmh.nii.gz")):
        rel = subject.relative_to(src).parts
        is_test = any(p.lower() == "test" for p in rel)
        # e.g. test/Amsterdam/Philips_VU .PETMR_01./168 -> Amsterdam_Philips_VU_PETMR_01_168
        parts = [
            re.sub(r"[^0-9A-Za-z]+", "_", p).strip("_")
            for p in rel
            if p.lower() not in ("training", "test")
        ]
        case = "_".join(parts)
        suffix = "Ts" if is_test else "Tr"
        flair = sitk.ReadImage(str(subject / "pre" / "FLAIR.nii.gz"))
        t1 = read_aligned(subject / "pre" / "T1.nii.gz", flair)
        label = sitk.Cast(read_aligned(subject / "wmh.nii.gz", flair), sitk.sitkUInt8)
        sitk.WriteImage(flair, str(out / f"images{suffix}" / f"{case}_0000.nii.gz"))
        sitk.WriteImage(t1, str(out / f"images{suffix}" / f"{case}_0001.nii.gz"))
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
    parser.add_argument("--acdc", type=Path, help="ACDC-2D-CL folder")
    parser.add_argument("--wmh", type=Path, help="WMH folder (searched recursively)")
    args = parser.parse_args()
    if args.acdc:
        convert_acdc(args.acdc)
    if args.wmh:
        convert_wmh(args.wmh)
