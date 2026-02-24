#!/usr/bin/env bash
source /home/z40575r/anaconda3/bin/activate
conda init bash
conda activate corrnet

MMPOSE_ROOT=/home/z40575r/anaconda3/envs/glofe/lib/python3.8/site-packages
python extract_openasl_cv_sequence_v3.py \
    $MMPOSE_ROOT/mmpose/.mim/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
    $MMPOSE_ROOT/mmpose/models/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    $MMPOSE_ROOT/mmpose/.mim/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark.py \
    $MMPOSE_ROOT/mmpose/models/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth \
    --sid $1 \
    --splits $2 \
    --device cuda:$3


# Model urls
# https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
# https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth
