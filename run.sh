#!/usr/bin/env bash
#
#SBATCH --job-name negative-sampling-in-data-loader
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

# create virtual env
python3 -m venv /tmp/venv/pykeen
source /tmp/venv/pykeen/bin/activate
pip install -U pip setuptools wheel

# install torch first
# CPU
# pip install torch==1.10.2+cpu torchvision==0.11.3+cpu torchaudio==0.10.2+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

# CUDA 11.3
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# install this repo's requirements
pip install -r requirements.txt

# master
pip install git+https://github.com/pykeen/pykeen.git@master
python main.py --top 4 --branch-name master

# branch
pip install git+https://github.com/pykeen/pykeen.git@negative-sampling-in-data-loader
python main.py --top 4 --branch-name negative-sampling-in-data-loader

# compare
# python compare.py
#git commit --all -m "Ran benchmark"
#git push
