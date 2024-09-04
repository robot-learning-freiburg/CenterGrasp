"""Install script for setuptools."""

import setuptools
from os import path

# read the contents of your README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="centergrasp",
    version="0.0.1",
    author="Eugenio Chisari",
    author_email="chisari@cs.uni-freiburg.de",
    install_requires=[
        "setuptools==59.5.0",
        "numpy==1.23.0",
        "open3d==0.18.0",
        "trimesh==3.16.4",
        "spatialmath-python==1.0.5",
        "h5py==3.10.0",
        "tqdm==4.66.1",
        "opencv-python==4.8.1.78",
        "Pillow==10.0.1",
        "pytorch-lightning==2.1.0",
        "wandb==0.15.12",
        "black==23.10.1",
        "zstandard==0.21.0",
        "colour==0.1.5",
        "scikit-image==0.19.3",
        "fvcore==0.1.5.post20221221",
        "scikit-learn==1.3.2",
        "tyro==0.5.10",
        "shortuuid==1.0.11",
        "roma==1.4.1",
        "opentsne==1.0.0",
        "pyrealsense2==2.54.2.5684",
        "rerun-sdk==0.10.1",
        "pycocotools==2.0.7",
        "sapien==2.2.2",
        "mplib==0.0.9",
        "urdfpy @ git+ssh://git@github.com/chisarie/urdfpy#egg=urdfpy",
        "hdfdict @ git+ssh://git@github.com/chisarie/hdfdict#egg=hdfdict",
        "mesh_to_sdf @ git+ssh://git@github.com/chisarie/mesh_to_sdf#egg=mesh_to_sdf",
    ],
    description="A grasping net",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # license="MIT",
    # url="https://github.com/chisarie/jax-agents",
    packages=setuptools.find_packages(),
    # classifiers=[
    #     "Programming Language :: Python :: 3",
    #     "License :: OSI Approved :: MIT License",
    #     "Operating System :: OS Independent",
    # ],
    python_requires=">=3.7",
)
