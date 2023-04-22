import os

from setuptools import setup

requires = open("./requirements.txt", "r").readlines() if os.path.exists("./requirements.txt") else open("./markit_rs_recorder.egg-info/requires.txt", "r").readlines()

setup(
    name="markit-rs-recorder",
    version="2.0.0",
    author="davidliyutong",
    author_email="davidliyutong@sjtu.edu.cn",
    description="Realsense remote recorder",
    packages=[
        "markit_rs_recorder",
        "markit_rs_recorder/cmd",
        "markit_rs_recorder/common",
        "markit_rs_recorder/filters",
        "markit_rs_recorder/io",
        "markit_rs_recorder/scripts",
    ],
    python_requires=">=3.6",
    install_requires=requires,
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown"
)
