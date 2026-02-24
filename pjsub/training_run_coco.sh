#!/usr/bin/env bash
source /home/z40575r/anaconda3/bin/activate
conda init bash
conda activate corrnet

python /home/z40575r/anaconda3/envs/corrnet/lib/python3.8/site-packages/mmpose/.mim/tools/train.py /home/z40575r/anaconda3/envs/corrnet/lib/python3.8/site-packages/mmpose/.mim/configs/wholebody/2d_kpt_sview_rgb_img/deeppose/coco-wholebody/res18_coco_wholebody_224x224.py --work-dir ./baseline/ --gpus 1 --local_rank 0
