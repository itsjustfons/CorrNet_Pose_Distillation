#!/usr/bin/env bash
source /home/z40575r/anaconda3/bin/activate
conda init bash
conda activate corrnet

python /data/group1/z40575r/CorrNet_pose_distillation/CorrNet/main.py --config /data/group1/z40575r/CorrNet_pose_distillation/CorrNet/configs/baseline3.yaml --device 0

