import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from N_bodyDP import pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    relic = pipeline.relic_stage(args.output, True)
    pipeline.figures_stage(args.output, relic, args.quick)


if __name__ == "__main__":
    main()
