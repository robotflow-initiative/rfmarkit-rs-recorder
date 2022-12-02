from realsense_recorder.cmd.local_capture_static_interactive import main


class DefaultArgs:
    config: str = "./realsense_config.yaml"
    tag: str = None


main(DefaultArgs)
