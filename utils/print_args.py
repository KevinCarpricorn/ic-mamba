"""Utility to pretty-print argparse arguments."""

def print_args(args):
    for k, v in sorted(vars(args).items()):
        print(f"{k}: {v}")
