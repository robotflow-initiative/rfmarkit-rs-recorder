from markit_rs_recorder.cmd.local_record_seq import main


class DefaultArgs:
    config: str = "./realsense_config.yaml"
    tag: str = None


main(DefaultArgs)
