"""
json_to_csv.py

Convert a pipeline output JSON (list of policy dicts) to CSV.

Usage:
    python json_to_csv.py --input ./output/LasVegas_policies.json
    python json_to_csv.py --input ./output/LasVegas_policies.json --output ./output/LasVegas_policies.csv
    python json_to_csv.py -i ./output/LasVegas_policies.json -o ./results/lv.csv

If --output is omitted the CSV is written next to the input file with the same stem.
"""

import argparse
import json
import os

import pandas as pd


def json_to_csv(input_path: str, output_path: str | None = None) -> str:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected a JSON array at the top level.")

    df = pd.DataFrame(data)

    if output_path is None:
        stem = os.path.splitext(input_path)[0]
        output_path = f"{stem}.csv"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a LEAP pipeline JSON output to CSV."
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to the input JSON file (list of policy dicts).",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Path for the output CSV. Defaults to <input_stem>.csv in the same directory.",
    )
    args = parser.parse_args()

    out = json_to_csv(args.input, args.output)
    print(f"Saved: {out}")
