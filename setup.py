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
        "numpy",
        "open3d",
        "trimesh",
        "spatialmath-python==1.0.5",
        "h5py",
        "tqdm",
        "opencv-python",
        "Pillow",
        "pytorch-lightning",
        "wandb",
        "black",
        "zstandard",
        "colour",
        "scikit-image",
        "fvcore",
        "scikit-learn",
        "tyro",
        "shortuuid",
        "roma",
        "opentsne",
        "pyrealsense2",
        "rerun-sdk",
        "pycocotools",
        "sapien",
        "mplib",
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
