#!bin/bash

# activate conda environment
eval "$(conda shell.bash hook)"
conda activate react

python train.py --batch-size 4  --gpu-ids 0,1  -lr 0.00001  --kl-p 0.00001 -e 2  -j 6  --outdir results/train_offline_50_epoch --dataset-path /home/tien/playground_facereconstruction/data/react_2024
