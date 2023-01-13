import argparse
import logging

import yaml
from rich.console import Console

from realsense_recorder.common import configure_realsense_system_from_keyboard


def main(args: argparse.Namespace):
    if args.input is not None:
        with open(args.input) as f:
            base_cfg_dict = yaml.load(f, Loader=yaml.SafeLoader)
    else:
        base_cfg_dict = None

    try:
        console = Console()
        res = configure_realsense_system_from_keyboard()
        if base_cfg_dict is not None:
            res.update(base_cfg_dict)
        with open(args.output, 'w') as f:
            yaml.dump(res, f)
        console.log(f"dump configuration to [bold green]{args.output}[/bold green]")
    except KeyboardInterrupt as e:
        logging.info("abort")

def entry_point(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, default=None, help="path to input config file")
    parser.add_argument("--output", "-o", type=str, default="realsense_config.yaml", help="path to output config file")
    args = parser.parse_args(argv)
    main(args)


if __name__ == '__main__':
    import sys
    entry_point(sys.argv)
