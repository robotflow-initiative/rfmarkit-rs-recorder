import sys

# NOTES to YUTONG:
# New dependencies
from realsense_recorder.filters.fiducial import detect_fiducial_marker

# pip install pupil_apriltags -- works on Ubuntu but not on Windows
# Search "MAGIC NUMBER" for magic numbers


if __name__ == '__main__':
    argv = sys.argv[1:]
    detect_fiducial_marker(
        argv[0],
        marker_length_m=0.018,
        enabled_cameras=['r08', 'r85'],
        april_tag_family="tag25h9",
        sequence_length=10,
        debug=True
    )
