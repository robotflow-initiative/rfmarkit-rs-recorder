import argparse
import sys
import realsense_recorder.scripts as scripts
import realsense_recorder.cmd  as cmd

parser = argparse.ArgumentParser()

args = sys.argv[1:]
if len(args) == 0:
    exit(print("No arguments provided"))
if args[0] == "configure":
    exit(scripts._EXPORTS[args[0]](args[1:]))
elif args[0] == "run":
    parser.add_argument("--app", type=str, default="local_capture_static_interactive", choices=["local_capture_static", "local_record_seq", "remote_record_seq"])
    run_args = parser.parse_args(args[1:])
    exit(cmd._EXPORTS[run_args.app](args[1:]))
else:
    print("Unknown command: {}".format(args[0]))
