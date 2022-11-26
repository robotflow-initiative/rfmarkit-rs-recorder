from realsense_recorder.common.record import new_realsense_camera_system_from_yaml_file

p = new_realsense_camera_system_from_yaml_file("./realsense_config.yaml")

print(p)