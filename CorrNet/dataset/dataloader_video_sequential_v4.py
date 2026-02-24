'''
This version of the data loader feeds a non-sequential input. It uses all video frames. If the length of a video is longer than max_frames, 
it loads the into the next batch
'''
import os
import cv2
import sys
import pdb
import six
import glob
import time
import torch
import random
import pandas as pd
import warnings
import pickle
from mmpose.datasets.pipelines.top_down_transform import TopDownGenerateTarget
import matplotlib.pyplot as plt
import bisect, math
#from mmpose.codecs import build_keypoint_codec

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
# import pyarrow as pa
from PIL import Image
import torch.utils.data as data
import matplotlib.pyplot as plt
from utils import video_augmentation
from torch.utils.data.sampler import Sampler

sys.path.append("..")
global kernel_sizes 

class BaseFeeder(data.Dataset):
    #EDIT: Fix so that it loads the ground truth pose heatmap instead
    def __init__(self, prefix, gloss_dict, dataset='phoenix2014', drop_ratio=1, num_gloss=-1, mode="train", transform_mode=True,
                datatype="lmdb", frame_interval=1, image_scale=1.0, kernel_size=1, input_size=224):
        self.mode = mode
        self.ng = num_gloss
        self.prefix = prefix
        self.dict = gloss_dict
        self.data_type = datatype
        self.dataset = dataset
        self.input_size = input_size
        global kernel_sizes 
        kernel_sizes = kernel_size
        self.frame_interval = frame_interval
        self.image_scale = image_scale
        self.feat_prefix = "/data/group1/z40575r/GloFE/cropped-vid/"
        self.pose_prefix = "/data/group1/z40575r/GloFE/tools/openasl_mmpose2/openasl_mmpose/"
        self.transform_mode = "train" if transform_mode else "test"
        self.max_frames = 256

        # Load metadata
        split = mode
        data_frame = pd.read_csv('/data/group1/z40575r/GloFE/openasl-v1.0.tsv', sep='\t')
        data_frame = data_frame.loc[data_frame['split'].str.contains(mode)]

        # --- Filter out missing/broken videos ---
        def filter_missing(row):
            path1 = os.path.join(self.pose_prefix, f'{row["vid"]}.pkl')
            full_path = path1.replace(':', '-')
            return os.path.exists(full_path) and os.path.getsize(full_path) > 0

        is_valid = data_frame.apply(filter_missing, axis=1)
        df_filtered = data_frame[is_valid]
        broken_videos = data_frame[~is_valid]['vid'].tolist()

        print(f"Split: {split}\nBefore filtering: {len(data_frame)}\nAfter filtering: {len(df_filtered)}")

        # Translation labels and sample names
        self.video_names = df_filtered['vid'].to_list()

        lengths_df = pd.read_csv("/data/group1/z40575r/CorrNet_pose_distillation/pose_lengths.tsv", sep="\t")
        vid2len = dict(zip(lengths_df["vid"], lengths_df["length"]))
        self.pose_lengths = vid2len

        self.cumu_segments = []
        self.video_index = []
        total_segments = 0
        skipped_steps = []

        for vid in self.video_names:
            if vid not in vid2len:
                continue
            length = vid2len[vid]
            num_segments = math.ceil(length / self.max_frames)

            self.video_index.append((vid, length, num_segments))
            total_segments += num_segments
            self.cumu_segments.append(total_segments)

        # --- Track which steps would have belonged to broken videos ---
        if broken_videos:
            cumu = 0
            for vid in data_frame['vid']:
                if vid in broken_videos:
                    if vid in vid2len:
                        length = vid2len[vid]
                        num_segments = math.ceil(length / self.max_frames)
                        broken_steps = list(range(cumu, cumu + num_segments))
                        skipped_steps.extend(broken_steps)
                        cumu += num_segments
                    else:
                        cumu += 1  # assume 1 segment if length unknown
                else:
                    if vid in vid2len:
                        cumu += math.ceil(vid2len[vid] / self.max_frames)
                    else:
                        cumu += 1

            print(f"⚠️ Warning: {len(broken_videos)} broken videos removed.")
            print(f"   Skipped step indices: {skipped_steps[:30]}{'...' if len(skipped_steps) > 30 else ''}")

        self.total_segments = total_segments
        self.inputs_list = np.array(self.video_names, dtype=object)
        print(mode, len(self))

        self.data_aug = self.transform()

        self.ann_info = {
            'num_joints': 133,
            'image_size': np.array((224, 224)),
            'heatmap_size': np.array((224, 224)),
            'joint_weights': False,
            'use_different_joint_weights': False
        }


        self.heatmap_transform = TopDownGenerateTarget(
            encoding='MSRA',
            sigma=8, #originally 4 , worked with 6 
            target_type='GaussianHeatmap'
        )

    def __getitem__(self, idx):
        if self.data_type == "video":
            # Figure out which video + segment this idx belongs to
            vid_idx = bisect.bisect_right(self.cumu_segments, idx)
            vid, length, num_segments = self.video_index[vid_idx]

            # Which segment inside this video?
            if vid_idx == 0:
                seg_idx = idx
            else:
                seg_idx = idx - self.cumu_segments[vid_idx - 1]

            start = seg_idx * self.max_frames
            end = min(start + self.max_frames, length)

            #Check for zero length segments
            if start >= length:
                start = max(0, length, self.max_frames)
                end = length
            
            if end <= start:
                print(f"[Warning] Empty segment detected for video {vid} (len={length})")
                end = min(length, start + 1)

            # --- Load frames + poses (delegate to read_video) --- #
            video_data, pose_output, file_id = self.read_video(vid, start, end)

            # --- Lookup pose length from .tsv --- #
            pose_length = self.pose_lengths[file_id]

            # --- Normalize --- #
            video_data, pose_output, pose_weights = self.normalize(video_data, pose_output)
            #breakpoint()
            return video_data, pose_output, pose_length, file_id, pose_weights


    def read_video(self, vid, start, end):
        video_path = os.path.join(self.feat_prefix, f"{vid}.mp4").replace(':', '-')
        
        # --- Read video frames --- #
        cap = cv2.VideoCapture(video_path)
        frame_list, frame_count = [], 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if start <= frame_count < end:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_list.append(frame_rgb)
            frame_count += 1
            if frame_count >= end:
                break
        cap.release()

        # --- Read pose segment --- #
        pose_output = self.read_pose_files(vid, start, end)

        # --- Safety checks --- #
        if len(frame_list) == 0:
            print(f"[WARN] Video '{vid}' returned 0 frames for segment {start}:{end}.")
            # Create 1 dummy black frame (H=224, W=224, C=3)
            frame_list = [np.zeros((224, 224, 3), dtype=np.uint8)]
            # Match pose to 1 frame
            pose_output = np.zeros((1, pose_output.shape[1])) if pose_output.ndim == 2 else np.zeros((1, *pose_output.shape[1:]))

        # --- Crop to equal length (safety check) --- #
        min_len = min(len(frame_list), pose_output.shape[0])
        if len(frame_list) != pose_output.shape[0]:
            print(f"[WARN] Frame/pose mismatch for '{vid}' — frames: {len(frame_list)}, poses: {pose_output.shape[0]} (cropping to {min_len})")

        frame_list = frame_list[:min_len]
        pose_output = pose_output[:min_len]

        # --- Extra sanity check --- #
        if not isinstance(frame_list[0], np.ndarray) or frame_list[0].ndim != 3:
            print(f"[WARN] Invalid frame format in '{vid}' at segment {start}:{end}. Using zero-frame fallback.")
            frame_list = [np.zeros((224, 224, 3), dtype=np.uint8)]
            pose_output = np.zeros_like(pose_output[:1])

        return frame_list, pose_output, vid



    def read_pose_files(self, vid, start, end):
        file_path = os.path.join(self.pose_prefix, f"{vid}.pkl").replace(':','-')
        with open(file_path, 'rb') as f:
            pose_keypoints = pickle.load(f)  # [T, 133, 2] or [T, 133, 3]

        # Slice only the requested segment
        pose_segment = pose_keypoints[start:end]

        return pose_segment


    def normalize(self, video, label, file_id=None):
        video, label = self.data_aug(video, label, file_id)
        video = video.float() / 127.5 - 1
        label_seq = []
        label_mask_seq = []
        #For each frame in label, do the following loop
        for frame in label:
            keypoints = frame
            #print(keypoints.shape)
            #Getting keypoints_visible
            visibility_flag = (keypoints[:,2] > 0.5).astype(np.float32)
            joints_3d_visible = np.zeros_like(keypoints)
            joints_3d_visible[:, 0] = visibility_flag

            #getting dataset keypoint weights
            input_dict = dict(
                joints_3d = keypoints,
                joints_3d_visible = joints_3d_visible,
                ann_info = self.ann_info
            )

            label_frame = self.heatmap_transform(input_dict)
            
            label_seq.append(label_frame['target'])
            label_mask_seq.append(label_frame['target_weight'])
        return video, label, label_mask_seq

    #for visualizing training label heatmaps
    def save_aggregated_heatmap(self, label, output_path='/data/group1/z40575r/CorrNet_pose_distillation/keypoint_heatmaps_aggregated_dev.png'):
        label = torch.tensor(label)  # Ensure it's a tensor

        aggregated = label.sum(dim=0)  # Shape: (72, 96)
        aggregated = aggregated / aggregated.max()  # Normalize to [0, 1]

        plt.imshow(aggregated.cpu().numpy(), cmap='hot', interpolation='nearest')
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()


    def save_individual_heatmaps(self, label, output_dir='/data/group1/z40575r/CorrNet_pose_distillation/keypoint_heatmaps/'):
        os.makedirs(output_dir, exist_ok=True)
        label = torch.tensor(label)  # Ensure it's a tensor

        for i in range(label.shape[0]):
            heatmap = label[i]  # Shape: (72, 96)

            plt.imshow(heatmap.cpu().numpy(), cmap='hot', interpolation='nearest')
            plt.axis('off')
            plt.savefig(os.path.join(output_dir, f'heatmap_{i:03d}.png'), bbox_inches='tight', pad_inches=0)
            plt.close()

    def transform(self):
        if self.transform_mode == "train":
            print("Apply training transform.")
            #pose to pose removes temporal rescale
            return video_augmentation.Compose([
                # video_augmentation.CenterCrop(224),
                # video_augmentation.WERAugment('/lustre/wangtao/current_exp/exp/baseline/boundary.npy'),
                #video_augmentation.RandomCrop(self.input_size), #probably no
                video_augmentation.RandomHorizontalFlip(0.5), #maybe yes
                #video_augmentation.Resize(self.image_scale), #maybe yes
                video_augmentation.ToTensor(),
                #video_augmentation.TemporalRescale(0.2, self.frame_interval), #definately no
            ])
        else:
            print("Apply testing transform.")
            return video_augmentation.Compose([
                #video_augmentation.CenterCrop(self.input_size),
                #video_augmentation.Resize(self.image_scale),
                video_augmentation.ToTensor(),
            ])

    def byte_to_img(self, byteflow):
        unpacked = pa.deserialize(byteflow)
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        return img

    def rand_view_transform(X, agx, agy, s):
        if X.shape[-1] == 2:
            padding = np.zeros((X.shape[0], X.shape[1], 1))
            X = np.concatenate((X, padding), axis=2)
        agx = math.radians(agx)
        agy = math.radians(agy)
        Rx = np.asarray([[1,              0,             0],
                         [0,  math.cos(agx), math.sin(agx)],
                         [0, -math.sin(agx), math.cos(agx)]])

        Ry = np.asarray([[math.cos(agy), 0, -math.sin(agy)],
                         [0, 1,              0],
                         [math.sin(agy), 0,  math.cos(agy)]])

        Ss = np.asarray([[s, 0, 0],
                         [0, s, 0],
                         [0, 0, s]])

        X0 = np.dot(np.reshape(X, (-1, 3)), np.dot(Ry, np.dot(Rx, Ss)))
        X = np.reshape(X0, X.shape)
        return X

    @staticmethod
    def collate_fn(batch):
        batch = [item for item in sorted(batch, key=lambda x: len(x[0]), reverse=True)]
        video, pose_output, pose_length, fi, pose_weights = list(zip(*batch))
        pose_output = [torch.stack([torch.from_numpy(f) for f in p], dim=0) for p in pose_output]
        pose_output = [torch.tensor(pose, dtype=torch.float32) for pose in pose_output]
        pose_weights = [torch.tensor(w, dtype=torch.float32) for w in pose_weights]
        #pose_weights = torch.stack(pose_weights, dim=0)
        left_pad = 0
        last_stride = 1
        total_stride = 1
        global kernel_sizes 
        for layer_idx, ks in enumerate(kernel_sizes):
            if ks[0] == 'K':
                left_pad = left_pad * last_stride 
                left_pad += int((int(ks[1])-1)/2)
            elif ks[0] == 'P':
                last_stride = int(ks[1])
                total_stride = total_stride * last_stride
        if len(video[0].shape) > 3:
            max_len = len(video[0])
            #breakpoint()
            video_length = torch.tensor([len(v) for v in video], dtype=torch.long)
            #video_length = torch.LongTensor([np.ceil(len(vid) / total_stride) * total_stride + 2*left_pad for vid in video])
            right_pad = int(np.ceil(max_len / total_stride)) * total_stride - max_len + left_pad
            max_len = max_len + left_pad + right_pad
            pose_max_len = max([p.size(0) for p in pose_output])

            padded_video = [torch.cat(
                (
                    #vid[0][None].expand(left_pad, -1, -1, -1),
                    vid,
                    vid[-1][None].expand(pose_max_len - len(vid), -1, -1, -1),
                )
                , dim=0)
                for vid in video]
            padded_video = torch.stack(padded_video)
            #breakpoint()
            padded_pose = [torch.cat(
                (
                #pose[0][None].expand(left_pad, -1, -1), #-1
                pose,
                pose[-1][None].expand(pose_max_len - len(pose), -1, -1), #-1
                ),dim=0)
                for pose in pose_output
            ]
            padded_pose_weights = [torch.cat(
                (
                    #pw[0][None].expand(left_pad, -1, -1),
                    pw,
                    pw[-1][None].expand(pose_max_len - pw.size(0), -1, -1)
                ),dim=0)
                for pw in pose_weights
            ]
            padded_pose_weights = torch.stack(padded_pose_weights)
        else:
            max_len = len(video[0])
            video_length = torch.LongTensor([len(vid) for vid in video])
            padded_video = [torch.cat(
                (
                    vid,
                    vid[-1][None].expand(max_len - len(vid), -1),
                )
                , dim=0)
                for vid in video]
            padded_video = torch.stack(padded_video).permute(0, 2, 1)
        pose_length = video_length
        #breakpoint()
        return padded_video, video_length, padded_pose, pose_length, padded_pose_weights

    def __len__(self):
        return self.total_segments

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time


if __name__ == "__main__":
    feeder = BaseFeeder()
    dataloader = torch.utils.data.DataLoader(
        dataset=feeder,
        batch_size=1,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
#    for data in dataloader:
#        pdb.set_trace()
