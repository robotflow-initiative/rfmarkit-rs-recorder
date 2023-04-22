from markit_rs_recorder.cmd.calibrate import main


class DefaultArgs:
    config: str = "./realsense_config.yaml"
    tag: str = None


main(DefaultArgs)
