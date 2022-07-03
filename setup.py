from setuptools import setup, find_packages

requirements = [
    'flask',
    'gevent',
]

setup(
    name="tcpbroker",
    version="1.3",
    author="davidliyutong",
    author_email="davidliyutong@sjtu.edu.cn",
    description="Realsense remote recorder",
    packages=["realsense_remote",],
    python_requires=">=3.6",
    install_requires=requirements,
    entrypoints={
        'console_scripts': [
            'realsense_remote = realsense_remote.main:main'
        ]
    }
)
