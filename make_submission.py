#!/usr/bin/env python3
"""Build <team>_submission.zip in the layout the challenge requires.

    <team>_submission.zip
    ├── output/matching_results.tsv, output/candidate_pairs.tsv
    ├── code/business_entity_resolution/  (src/, README.md, requirements.txt)
    └── Documentation_template.md

Runs the official validator on the two output files first and refuses to package on a
failure. Get the output files of the chosen run from S3 first, for example:
    aws s3 cp s3://amazon-ml-challenge-26-sagemaker-125650147728/predictions/M-v3/ output/ --recursive

    python make_submission.py --team "Team Name"
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "code" / "business_entity_resolution"
OUTPUTS = ("matching_results.tsv", "candidate_pairs.tsv")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True, help="team name, used for the zip file name")
    ap.add_argument("--output-dir", default=str(ROOT / "output"))
    ap.add_argument("--test-dir", default=str(ROOT / "resources" / "student_resource" / "dataset" / "test"))
    args = ap.parse_args()

    out = Path(args.output_dir)
    missing = [f for f in OUTPUTS if not (out / f).exists()]
    if missing:
        print(f"missing in {out}: {', '.join(missing)}")
        return 1
    if "[Your Team Name]" in (ROOT / "Documentation_template.md").read_text(encoding="utf-8"):
        print("warning: Documentation_template.md still has the team-name placeholder")

    validator = ROOT / "resources" / "student_resource" / "utils" / "validate_submission.py"
    r = subprocess.run([sys.executable, str(validator), "--matching", str(out / OUTPUTS[0]),
                        "--candidate", str(out / OUTPUTS[1]), "--test-dir", args.test_dir])
    if r.returncode != 0:
        print("validator failed: not packaging")
        return 1

    zip_path = ROOT / f"{args.team.strip().replace(' ', '_')}_submission.zip"
    skip = lambda p: "__pycache__" in p.parts or p.suffix == ".pyc"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in OUTPUTS:
            z.write(out / f, f"output/{f}")
        for p in sorted(CODE.rglob("*")):
            if p.is_file() and not skip(p):
                z.write(p, f"code/business_entity_resolution/{p.relative_to(CODE).as_posix()}")
        z.write(ROOT / "Documentation_template.md", "Documentation_template.md")
    names = zipfile.ZipFile(zip_path).namelist()
    print(f"wrote {zip_path.name}: {len(names)} files, {zip_path.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
