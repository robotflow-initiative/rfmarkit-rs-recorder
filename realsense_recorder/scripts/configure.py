import argparse

import yaml
from rich.console import Console

from realsense_recorder.common import configure_realsense_system_from_keyboard


def main(args):
    try:
        console = Console()
        res = configure_realsense_system_from_keyboard()
        with open(args.output, 'w') as f:
            yaml.dump(res, f)
        console.log(f"dump configuration to [bold green]{args.output}[/bold green]")
    except KeyboardInterrupt as e:
        console.log("abort")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", type=str, default="realsense_config.yaml")
    args = parser.parse_args()
    main(args)
