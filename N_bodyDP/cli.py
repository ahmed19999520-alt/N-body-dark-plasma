import argparse
import json

from .pipeline import run


def main(argv=None):
    parser = argparse.ArgumentParser(prog="nbodydp-pipeline")
    parser.add_argument("--output", default="results")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--n-live", type=int, default=300)
    parser.add_argument("--skip-bayesian", action="store_true")
    args = parser.parse_args(argv)
    summary = run(args.output, quick=args.quick, n_live=args.n_live, skip_bayesian=args.skip_bayesian)
    print(json.dumps(summary["verification"], indent=2))


if __name__ == "__main__":
    main()
