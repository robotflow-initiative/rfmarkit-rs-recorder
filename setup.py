from setuptools import setup

requirements = [
    'flask',
    'gevent',
    'pyrealsense2',
    'opencv-python',
    'tqdm',
    'numpy',
]

setup(
    name="realsense_recorder",
    version="1.3",
    author="davidliyutong",
    author_email="davidliyutong@sjtu.edu.cn",
    description="Realsense remote recorder",
    packages=[
        "realsense_recorder",
        "realsense_recorder/apps",
        "realsense_recorder/common",
        "realsense_recorder/io"
        "realsense_recorder/remote",
        "realsense_recorder/scripts",
    ],
    python_requires=">=3.6",
    install_requires=requirements,
    entrypoints={
        'console_scripts': [
            'remote = remote.main:main'
        ]
    }
)
