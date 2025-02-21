from setuptools import find_packages, setup

setup(
    name="luxai-s3-fast",
    version="0.0.0",
    packages=find_packages(),
    install_requires=[
        "luxai_s3",
    ],
    entry_points={"console_scripts": ["luxai-s3-fast = luxai_runner_fast.cli:main"]},
    author="Lux AI Challenge + Kohki Horie",
    description="Lux AI Challenge Season 3 environment code, but faster",
)
