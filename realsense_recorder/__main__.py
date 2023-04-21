import argparse
import sys
import realsense_recorder.scripts as scripts
import realsense_recorder.cmd as cmd

parser = argparse.ArgumentParser()

args = sys.argv[1:]
if len(args) == 0:
    exit(print("No arguments provided"))
if args[0] == "configure":
    exit(scripts.configure(args[1:]))
elif args[0] == "calibrate":
    exit(cmd.calibrate(args[1:]))
elif args[0] == "serve":
    exit(cmd.serve(args[1:]))
elif args[0] == "post_process":
    exit(cmd.post_process(args[1:]))
else:
    print("Unknown command: {}".format(args[0]))
