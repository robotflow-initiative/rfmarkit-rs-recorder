import os

from setuptools import setup

requires = open("./requirements.txt", "r").readlines() if os.path.exists("./requirements.txt") else open("./markit_realsense_recorder.egg-info/requires.txt", "r").readlines()

setup(
    name="markit-realsense-recorder",
    version="1.8.0",
    author="davidliyutong",
    author_email="davidliyutong@sjtu.edu.cn",
    description="Realsense remote recorder",
    packages=[
        "realsense_recorder",
        "realsense_recorder/cmd",
        "realsense_recorder/common",
        "realsense_recorder/filters",
        "realsense_recorder/io",
        "realsense_recorder/scripts",
    ],
    python_requires=">=3.6",
    install_requires=requires,
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown"
)
