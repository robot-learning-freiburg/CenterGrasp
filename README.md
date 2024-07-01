# CenterGrasp: Object-Aware Implicit Representation Learning for Simultaneous Shape Reconstruction and 6-DoF Grasp Estimation

Repository providing the source code for the paper "CenterGrasp: Object-Aware Implicit Representation Learning for Simultaneous Shape Reconstruction and 6-DoF Grasp Estimation", see the [project website](http://centergrasp.cs.uni-freiburg.de/). Please cite the paper as follows:

	@article{chisari2024centergrasp,
	  title={CenterGrasp: Object-Aware Implicit Representation Learning for Simultaneous Shape Reconstruction and 6-DoF Grasp Estimation},
	  shorttile={CenterGrasp},
	  author={Chisari, Eugenio and Heppert, Nick and Welschehold, Tim and Burgard, Wolfram and Valada, Abhinav},
	  journal={IEEE Robotics and Automation Letters (RA-L)},
	  year={2024}
	}

## Installation

For centergrasp

```bash
conda create --name centergrasp_g_env python=3.8
conda activate centergrasp_g_env
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
# pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu
pip install kaolin==0.14.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-1.13.1_cu117.html
git clone git@github.com:PRBonn/manifold_python.git
cd manifold_python
git submodule update --init
make install
cd centergrasp_g
pip install -e .
```

For GIGA

```bash
pip install cython
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
# pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.1+cpu.html
pip install catkin-pkg --extra-index-url https://rospypi.github.io/simple/
git clone git@github.com:chisarie/GIGA.git
cd GIGA
git checkout baseline
pip install -e .
python scripts/convonet_setup.py build_ext --inplace
```

## Data Download

Follow the instructions in the `README.md` of GIGA's repository (https://github.com/UT-Austin-RPL/GIGA) to download giga's pretrained models and object meshes, which are needed to run the evaluations.

## Pretrained Weights Download

Please download the pretrained weights ([sgdf decoder](http://centergrasp.cs.uni-freiburg.de/download/ckpt_sgdf/9vkd9370.zip), [rgb encoder](http://centergrasp.cs.uni-freiburg.de/download/ckpt_rgb/12c7ven5.zip)), extract them, and place them in the `ckpt_sgdf` and `ckpt_rgb` folders respectively, at the root of the repository. These are the models trained on the GIGA set of objects.

