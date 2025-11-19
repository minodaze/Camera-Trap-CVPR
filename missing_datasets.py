#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

DEFAULT_ROOT = "/fs/scratch/PAS2099/Lemeng/icicle/camera-trap-final/np_result_feature_saving_new_threshold"

DEFAULT_ITEMS = """
nz_nz_PS1_CAM6213
KGA_KGA_KHOLA03
nz_nz_EFH_HCAMB05
serengeti_serengeti_L06
KGA_KGA_KHOGA04
nz_nz_EFH_HCAMC02
KAR_KAR_B03
serengeti_serengeti_L10
nz_nz_EFH_HCAMD08
serengeti_serengeti_N04
serengeti_serengeti_Q10
serengeti_serengeti_V10
serengeti_serengeti_F08
nz_nz_EFH_HCAME08
serengeti_serengeti_T10
nz_nz_EFH_HCAME09
serengeti_serengeti_S11
serengeti_serengeti_H03
serengeti_serengeti_H11
serengeti_serengeti_K11
ENO_ENO_C02
serengeti_serengeti_Q09
ENO_ENO_C04
na_na_lebec_CA-37
nz_nz_EFD_DCAMF06
nz_nz_EFH_HCAMI01
serengeti_serengeti_D02
serengeti_serengeti_E05
serengeti_serengeti_D09
ENO_ENO_D06
ENO_ENO_E06
MAD_MAD_D04
serengeti_serengeti_Q07
nz_nz_EFH_HCAMC03
nz_nz_PS1_CAM7312
serengeti_serengeti_Q11
serengeti_serengeti_R10
caltech_caltech_88
nz_nz_EFH_HCAMF01
serengeti_serengeti_E12
serengeti_serengeti_H08
MAD_MAD_H08
na_na_lebec_CA-18
nz_nz_EFH_HCAME05
caltech_caltech_46
idaho_idaho_122
nz_nz_PS1_CAM8008
serengeti_serengeti_O13
wellington_wellington_031c
MAD_MAD_C07
MTZ_MTZ_D03
nz_nz_EFH_HCAMB01
APN_APN_K051
KAR_KAR_A01
MAD_MAD_B03
APN_APN_TB17
MAD_MAD_A04
MAD_MAD_B06
nz_nz_EFD_DCAMH07
caltech_caltech_38
CDB_CDB_A05
na_na_lebec_CA-19
na_na_lebec_CA-21
nz_nz_EFH_HCAMG13
nz_nz_EFD_DCAMD10
PLN_PLN_B04
APN_APN_13U
APN_APN_N1
MTZ_MTZ_E05
MTZ_MTZ_D06
na_na_lebec_CA-31
nz_nz_EFD_DCAMH01
APN_APN_U43B
na_na_lebec_CA-05
nz_nz_EFD_DCAMG03
APN_APN_U23A
APN_APN_WM
caltech_caltech_70
APN_APN_K082
MTZ_MTZ_F04
""".strip().splitlines()


def load_items(path: Path | None) -> list[str]:
    if path is None:
        return [s.strip() for s in DEFAULT_ITEMS if s.strip()]
    txt = path.read_text(encoding="utf-8")
    lines = []
    for line in txt.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # allow users to paste noisy rows; take the first whitespace-separated token
        line = line.split()[0]
        lines.append(line)
    return lines


def to_result_filename(dataset: str) -> str:
    """
    Convert 'APN/APN_13U' -> 'APN_APN_13U_per_ckp.csv'
    """
    return dataset.replace("/", "_") + "_per_ckp.csv"


def main():
    ap = argparse.ArgumentParser(description="List datasets missing their _per_ckp.csv results.")
    ap.add_argument("--root", default=DEFAULT_ROOT, help="Results root directory")
    ap.add_argument("--list", type=Path, default=None, help="Optional file with one dataset/subset per line")
    ap.add_argument("--out", type=Path, default=None, help="Optional path to write missing list")
    ap.add_argument("--print-paths", action="store_true",
                    help="Print full expected file paths instead of dataset names")
    args = ap.parse_args()

    root = Path(args.root)
    items = load_items(args.list)

    missing_ds: list[str] = []
    for ds in items:
        expected = root / to_result_filename(ds)
        if not expected.is_file():
            missing_ds.append(ds)

    # Output
    if args.print_paths:
        for ds in missing_ds:
            print(str(root / to_result_filename(ds)))
    else:
        for ds in missing_ds:
            print(ds)

    # Optional write-out
    if args.out:
        args.out.write_text("\n".join(missing_ds) + "\n", encoding="utf-8")

    # Summary to stderr
    print(f"# Missing: {len(missing_ds)} / Total: {len(items)}", file=sys.stderr)


if __name__ == "__main__":
    main()
