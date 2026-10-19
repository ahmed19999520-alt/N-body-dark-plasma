import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from N_bodyDP import pipeline
from N_bodyDP.verification import render_markdown, run_all_checks, summarise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results")
    parser.add_argument("--include-calibrated-relic", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    pipeline.octet_stage(args.output)
    pipeline.self_interaction_stage(args.output)
    pipeline.gw_stage(args.output)
    pipeline.other_stage(args.output)
    checks = run_all_checks(args.include_calibrated_relic)
    text = render_markdown(checks)
    with open(os.path.join(args.output, "verification.md"), "w") as fh:
        fh.write(text)
    print(text)
    print(summarise(checks))


if __name__ == "__main__":
    main()
