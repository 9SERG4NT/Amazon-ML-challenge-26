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

For another pipeline's package (e.g. the team's bi-encoder run), point --output-dir, --code-dir and --doc at its files:
    python make_submission.py --team "Team Name" --output-dir <dir with both TSVs> --code-dir <its code folder> --doc <its filled template>
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
    ap.add_argument("--code-dir", default=str(CODE), help="packed as code/business_entity_resolution/")
    ap.add_argument("--doc", default=str(ROOT / "Documentation_template.md"), help="the filled methodology template")
    args = ap.parse_args()

    out, code, doc = Path(args.output_dir), Path(args.code_dir), Path(args.doc)
    missing = [str(out / f) for f in OUTPUTS if not (out / f).exists()]
    missing += [str(p) for p in (code / "src", code / "README.md", code / "requirements.txt", doc) if not p.exists()]
    if missing:
        print(f"missing: {', '.join(missing)}")
        return 1
    if "[Your Team Name]" in doc.read_text(encoding="utf-8"):
        print(f"warning: {doc.name} still has the team-name placeholder")

    validator = ROOT / "resources" / "student_resource" / "utils" / "validate_submission.py"
    r = subprocess.run([sys.executable, str(validator), "--matching", str(out / OUTPUTS[0]),
                        "--candidate", str(out / OUTPUTS[1]), "--test-dir", args.test_dir])
    if r.returncode != 0:
        print("validator failed: not packaging")
        return 1

    zip_path = ROOT / f"{args.team.strip().replace(' ', '_')}_submission.zip"
    skip = lambda p: "__pycache__" in p.parts or "catboost_info" in p.parts or p.suffix == ".pyc"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in OUTPUTS:
            z.write(out / f, f"output/{f}")
        for p in sorted(code.rglob("*")):
            if p.is_file() and not skip(p):
                z.write(p, f"code/business_entity_resolution/{p.relative_to(code).as_posix()}")
        z.write(doc, "Documentation_template.md")
    names = zipfile.ZipFile(zip_path).namelist()
    print(f"wrote {zip_path.name}: {len(names)} files, {zip_path.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
