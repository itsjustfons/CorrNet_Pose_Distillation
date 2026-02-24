# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from argparse import ArgumentParser
import pdb
import inspect
import cv2
import mmcv
import mmpose.apis 
import mmpose.datasets
import time
import torch
from mmpose.core.evaluation.top_down_eval import _get_max_preds
from collections import OrderedDict

#MAC calculations
'''
import hydra
import torch
from omegaconf import DictConfig
from torchprofile import profile_macs
from thop import profile
'''

import importlib

from mmpose.apis import (collect_multi_frames, inference_top_down_pose_model,
                         init_pose_model, process_mmdet_results, vis_pose_result)
from mmpose.datasets import DatasetInfo
#from mmdet.apis import inference_detector, init_detector
#from mmcv import (inference_detector, init_detector)

#try:
#    from mmdet.apis import inference_detector, init_detector
#    has_mmdet = True
#except (ImportError, ModuleNotFoundError):
#    has_mmdet = False

from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
import numpy as np
import pickle as pkl

def import_class(name):
    components = name.rsplit('.', 1)
    mod = importlib.import_module(components[0])
    mod = getattr(mod, components[1])
    return mod

def load_model_weights(model, weight_path):
    state_dict = torch.load(weight_path)
    '''
    if len(self.arg.ignore_weights):
        for w in self.arg.ignore_weights:
            if state_dict.pop(w, None) is not None:
                print('Successfully Remove Weights: {}.'.format(w))
            else:
                print('Can Not Remove Weights: {}.'.format(w))
    '''
    weights = modified_weights(state_dict['model_state_dict'], False)
    #print(weights.keys())
    
    #weights = self.modified_weights(state_dict['model_state_dict'])
    #breakpoint()
    model.load_state_dict(weights, strict=False)
    return model

def modified_weights(state_dict, modified=False):
    state_dict = OrderedDict([(k.replace('.module', ''), v) for k, v in state_dict.items()])
    if not modified:
        return state_dict
    modified_dict = dict()
    return modified_dict

def load_corrnet():
    print("Loading model")
    model_class = import_class('slr_network_2head_v6.SLRModel') #ensure using correct model slr_network or slr_network2head
    model = model_class(c2d_type='resnet18', num_classes=1296, conv_type=2, )

    #shutil.copy2(inspect.getfile(model_class), self.arg.work_dir)

    model = load_model_weights(model, '/data/group1/z40575r/CorrNet_pose_distillation/CorrNet/work_dir/keypoint_regression_img_grad_accum_v18_max16_bs8_a1_b1/epoch0_step140000.pt')
    #move model to GPU 
    model = model.to(torch.device("cuda"))
    model.eval()
    print("Loading model finished.")

    return model

def main():
    """Visualize the demo video (support both single-frame and multi-frame).

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--video-path', type=str, help='Video path')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show visualizations.')
    parser.add_argument(
        '--out-video-root',
        default='',
        help='Root of the output video file. '
        'Default not saving the visualization video.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=1,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    parser.add_argument(
        '--use-multi-frames',
        action='store_true',
        default=False,
        help='whether to use multi frames for inference in the pose'
        'estimation stage. Default: False.')
    parser.add_argument(
        '--online',
        action='store_true',
        default=False,
        help='inference mode. If set to True, can not use future frame'
        'information when using multi frames for inference in the pose'
        'estimation stage. Default: False.')
    parser.add_argument(
        '--sid',
        type=int,
        default=0)
    parser.add_argument(
        '--splits',
        type=int,
        default=1)
	

    #print ("has_mmdet is")
    #print (has_mmdet)
    #assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    # assert args.show or (args.out_video_root != '')
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    print('Initializing model...')
    # build the detection model from a config file and a checkpoint file
    '''
    det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())
    '''

    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())
    

    # build pose modelfrom CorrNet
    corrnet_pose = load_corrnet()

    dataset = pose_model.cfg.data['test']['type']
    # get datasetinfo
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    arg_dict = {
        #'det_model': det_model,
        'pose_model': pose_model,
        'dataset': dataset,
        'dataset_info': dataset_info,
        'output_root': 'keypoints_pose_regression_img_v18_140k',
        'args': args,
    }

    all_samples = load_sample_names('../valid_test_vids.txt') #original code. Uncomment if not doing the MAC calculations
    #all_samples = load_sample_names('../single_vid.txt') #3.166 seconds, 77 frames. 
    # all_samples = all_samples[:5]
    total_samples = len(all_samples)
    print('Total samples:', total_samples)
    print(arg_dict['output_root'])
    chunk = (total_samples + args.splits - 1) // args.splits
    sample_split = all_samples[args.sid * chunk: min((args.sid + 1) * chunk, total_samples)]
    print(f'Running split:[{args.sid * chunk}:{min((args.sid + 1) * chunk, total_samples)}]')

    
    for sample_vid in tqdm(sample_split):
        sample_id = sample_vid.split('/')[-1][:-4]
        output_file_path = os.path.join(arg_dict['output_root'], f'{sample_id}.pkl')
        if os.path.exists(output_file_path):
            continue
        process_single_video(sample_vid, arg_dict, corrnet_model=corrnet_pose)
    
    
    #process_single_video("/data/group1/z40575r/TimeSformer_GloFE/TimeSformer/datasets/how2sign/256/_-adcxjm1R4_0-8-rgb_front.mp4", arg_dict, corrnet_model=corrnet_pose)
    


def process_single_video(video_path, arg_dict, corrnet_model):
    #start = time.time()
    output_root = arg_dict['output_root']
    args = arg_dict['args']
    # read video
    cap = cv2.VideoCapture(video_path)
    opencv_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR → RGB (match training preprocessing)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        opencv_frames.append(frame_rgb)
    cap.release()

    # return the output of some desired layers,
    # e.g. use ('backbone', ) to return backbone feature
    results = []
    sample_id = video_path.split('/')[-1][:-4]
    # print('Running inference...')
    #start = time.time()
    for frame_id, cur_frame in enumerate(mmcv.track_iter_progress(opencv_frames)):
        frame_tensor = torch.from_numpy(cur_frame).permute(2, 0, 1).float()
        frame_tensor = frame_tensor / 127.5 - 1.0 #normalize

        sequence_tensor = frame_tensor.unsqueeze(0).repeat(16, 1, 1, 1)
        batched_tensor = sequence_tensor.unsqueeze(0).to('cuda')
        #breakpoint()
        with torch.no_grad():
            heatmap = corrnet_model(batched_tensor)
            heatmap_np = heatmap['predict_heatmap'].cpu().numpy()
        
        #if len(pose_results) != 0:
        #breakpoint()
        if len(heatmap_np) != 0:
            #pose_results = pose_results[0]
            pose_results = heatmap_np[0]#[0]
            #results.append(pose_results['keypoints'])
            results.append(pose_results)
        else:
            print(f'{sample_id} Frame:{frame_id} has no person')
            with open(f'log-ext-openasl-s{args.sid}.txt', 'a') as f:
                f.write(f'{sample_id} Frame:{frame_id} has no person\n')
    results = np.array(results)
    #breakpoint()
    with open(os.path.join(output_root, f'{sample_id}.pkl'), 'wb') as f:
        if len(results.shape) == 3:
            pkl.dump(results, f)
        else:
            with open(f'log-ext-openasl-pvid-s{args.sid}.txt', 'a') as f:
                f.write(f'{sample_id} Incorrect result shape: {results.shape}\n')

def list_all_vid_names(root_dir, out_list_file):
    paths = []
    samples = sorted(os.listdir(root_dir))
    for sample in samples:
        paths.append(os.path.join(root_dir, sample))
    with open(out_list_file, 'w') as f:
        f.writelines('\n'.join(paths))

def load_sample_names(txt_path):
    with open(txt_path, 'r') as f:
        paths = f.readlines()
    paths = [x.strip() for x in paths]
    print('Total samples:', len(paths))
    return paths


if __name__ == '__main__':
    main()
    # list_all_vid_names('/mnt/workspace/OpenASL/video-clips', 'open_asl_samples.txt')
