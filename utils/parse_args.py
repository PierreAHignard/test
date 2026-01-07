# utils/parse_args.py
import argparse

__all__ = [
    "parse_arguments"
]

def parse_arguments():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description="Run CV pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--log-level', type=str, default='normal',
        choices=['silent', 'normal', 'verbose'],
        help='Logging level: silent (no output), normal (main steps), verbose (all details)'
    )

    parser.add_argument(
        '--run_name', type=str, default='',
        help='Name to give to the run. Defaults to general parameters and date'
    )

    return parser.parse_args()
