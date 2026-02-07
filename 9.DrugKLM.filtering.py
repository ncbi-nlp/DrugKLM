#!/usr/bin/env python3

import csv
import argparse

ALLOWED_CATEGORIES = {"11", "13", "15"}

def main(input_tsv, output_tsv):

    with open(input_tsv, newline="", encoding="utf-8") as fin, \
         open(output_tsv, "w", newline="", encoding="utf-8") as fout:

        reader = csv.DictReader(fin, delimiter="\t")
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames, delimiter="\t")
        writer.writeheader()

        for row in reader:
            if row.get("FDA-Approved") != "Yes":
                continue

            if row.get("Category #") not in ALLOWED_CATEGORIES:
                continue

            writer.writerow(row)

    print(f"Filtered output written to: {output_tsv}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Filter FDA-approved drugs with positive pre-clinical evidence"
    )

    # named arguments
    parser.add_argument("--input", required=True, help="Input Categorization TSV file")
    parser.add_argument("--output", required=True, help="Filtered output TSV file")

    # positional (optional, for backward compatibility)
    parser.add_argument("pos_input", nargs="?", help="Input TSV file")
    parser.add_argument("pos_output", nargs="?", help="Output TSV file")

    args = parser.parse_args()

    input_tsv = args.input or args.pos_input
    output_tsv = args.output or args.pos_output

    if not input_tsv or not output_tsv:
        parser.error("Both --input and --output are required")

    main(input_tsv, output_tsv)
