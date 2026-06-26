"""
Download the CMS ICD-10-CM code list and save it as data/raw/icd10_codes.csv.

CMS releases a new file each fiscal year. This script targets FY2025.
Run from the repo root:

    python scripts/fetch_icd10.py
"""

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

URL = (
    "https://www.cms.gov/files/zip/2025-code-descriptions-tabular-order.zip"
)

# Tab-delimited file inside the ZIP; columns are positional, not headered
INNER_FILE = "icd10cm_codes_2025.txt"
OUT = Path("data/raw/icd10_codes.csv")


def main() -> None:
    print(f"Downloading {URL} …")
    resp = requests.get(URL, timeout=60)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        names = zf.namelist()
        print("Files in ZIP:", names)
        target = next((n for n in names if n.endswith(INNER_FILE)), None)
        if target is None:
            raise FileNotFoundError(f"{INNER_FILE!r} not found in ZIP — available: {names}")

        raw = zf.read(target).decode("utf-8", errors="replace")

    # Format: fixed-width — code (up to 7 chars, no dot) padded with spaces,
    # then one or more spaces, then the long description. No header row.
    # e.g. "A000    Cholera due to Vibrio cholerae 01, biovar cholerae"
    records = []
    for line in raw.splitlines():
        line = line.rstrip()
        if not line:
            continue
        parts = line.split(None, 1)  # split on first whitespace
        if len(parts) == 2:
            code_raw, desc = parts
            # Insert decimal point per ICD-10 convention (after 3rd char)
            code = code_raw[:3] + ("." + code_raw[3:] if len(code_raw) > 3 else "")
            records.append({"code": code, "description": desc.strip()})

    df = pd.DataFrame(records)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df):,} codes to {OUT}")


if __name__ == "__main__":
    main()
