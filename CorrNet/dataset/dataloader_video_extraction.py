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
        self.frame_interval = frame_interval # not implemented for read_features()
        self.image_scale = image_scale # not implemented for read_features()
        self.feat_prefix = "/data/group1/z40575r/GloFE/cropped-vid/"
        self.pose_prefix = "/data/group1/z40575r/GloFE/tools/openasl_mmpose2/openasl_mmpose/"
        self.transform_mode = "train" if transform_mode else "test"
        #Loading GloFE data
        split = mode
        data_frame = pd.read_csv('/data/group1/z40575r/GloFE/openasl-v1.0.tsv', sep='\t') #-mini for checking sequences
        data_frame = data_frame.loc[data_frame['split'].str.contains(mode)]

        def filter_missing(row):
            path1 = os.path.join(self.pose_prefix, f'{row["vid"]}.pkl')
            full_path = path1.replace(':','-')
            #print(full_path)
            #full_path = os.path.join(self.feat_path, f'{row["vid"]}.pkl') #OG code
            #full_path = os.path[full_path.replace(':', '-')]
            #print(full_path)
            return os.path.exists(full_path) and os.path.getsize(full_path) > 0

        is_missing = data_frame.apply(filter_missing, axis=1)
        df_filtered = data_frame[is_missing]
        #if self.local_rank == 0:
        print(f'Split:{split}\nBefore filtering: {len(data_frame)}\n After filtering: {len(df_filtered)}')
        # translation labels and sample names (split agnostic)
        self.video_names = df_filtered['vid'].to_list()

        #self.inputs_list = np.load(f"./preprocess/{dataset}/{mode}_info.npy", allow_pickle=True).item()
        self.inputs_list = np.array(self.video_names, dtype=object)
        print(mode, len(self))
        self.data_aug = self.transform()
        #print("")

        codec_cfg = dict(
            encoding='MSRA',
            input_size=(256,256), #check the actual input size of the pose data
            heatmap_size=(96,72), #7,7 from original corrnet. 64,64 from mmpose demo. 96,72 from top_down model
            sigma = 2,
            target_type = 'GaussianHeatmap',
            use_udp = False, 
            #use_different_joint_weights = False
        )

        self.ann_info = {
            'num_joints' : 133, #when training do 133
            'image_size' : np.array((224,224)),
            'heatmap_size': np.array((48,64)), #(48,64), (96,72), (64,64), 
            'sigma': 2,
            'joint_weights': False,
            'use_different_joint_weights': False
        }

        self.heatmap_transform = TopDownGenerateTarget(
            encoding='MSRA',
            sigma=3, #originally 4 , worked with 6 
            target_type='GaussianHeatmap'
        )

    def __getitem__(self, idx): 
        if self.data_type == "video":
            #frame_list, pose_output, pose_length, fi
            video_data, pose_output, pose_length, fi = self.read_video(idx)
            video_data, pose_output, pose_weights = self.normalize(video_data, pose_output)
            #input_data, label = self.normalize(input_data, label, fi['fileid'])
            
            #have pose_output clone first frame
            #print("pose output shape", pose_output.shape)
            
            #HACK. Frame freezing. Make it toggle-able
            #first_pose_frame = pose_output[0].copy()
            #pose_output = np.broadcast_to(first_pose_frame, pose_output.shape)

            #Get only first 3 frames of input video data
            video_data = video_data[:3]
            #print(video_data.shape) torch.Size([3, 3, 224, 224])
            #print(video_data.dtype) torch.float32
            return video_data, pose_output, pose_length, fi, pose_weights
            #return video_data, torch.LongTensor(label), self.inputs_list[idx]['original_info']


    def read_video(self, index):
        # load file info
        fi = self.video_names[index]
        print(fi)
        video_path = os.path.join(self.feat_prefix, f"{fi}.mp4")
        video_path = video_path.replace(':','-')
        
        # read video frames using OpenCV
        cap = cv2.VideoCapture(video_path) #Try reading with mmcv instead?
        frame_list = []
        frame_count = 0
        offset = int(torch.randint(0, self.frame_interval, [1]))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if (frame_count - offset) % self.frame_interval == 0 and frame_count >= offset:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_list.append(frame_rgb)
            frame_count += 1
        cap.release()
        
        #generate label list. Change this to instead read pose data
        '''
        label_list = []
        for phase in fi['label'].split(" "):
            if phase == '':
                continue
            if phase in self.dict.keys():
                label_list.append(self.dict[phase][0])
        '''
        pose_output, pose_length = self.read_pose_files(index)
        #print(type(frame_list))

        return frame_list, pose_output, pose_length, fi


    def read_pose_files(self, index: int):
        # MMPose 76
        body_sample_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        face_sample_indices = [71, 77, 85, 89] + \
                              [40, 42, 44, 45, 47, 49] + \
                              [59, 60, 61, 62, 63, 64] + [65, 66, 67, 68, 69, 70] + \
                              [50]
        
        # read files
        vid_name = self.video_names[index]

        file2 = os.path.join(self.pose_prefix, f'{vid_name}.pkl') #added this to fix file name isue
        file_path = file2.replace(':','-')
        with open(file_path, 'rb') as f:
            pose_keypoints = pickle.load(f)  # T K(133) C
        T = pose_keypoints.shape[0]
        pose_length = T #if T <= self.visual_token_num else self.visual_token_num
        

        pose_output = pose_keypoints
        return pose_output, pose_length

    def normalize(self, video, label, file_id=None):
        video, label = self.data_aug(video, label, file_id)
        video = video.float() / 127.5 - 1
        #Transform pose data. Needs to be some kind of dictionary {keypoints, keypoints_visible, dataset keypoint weights}
        #Getting keypoints
        keypoints = label[0]

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

        label = self.heatmap_transform(input_dict)
        #breakpoint()
        label_mask = label['target_weight']
        label = label['target']
        #print("label keys",label.keys())
        #print("target weight:", label_mask)
        
        #breakpoint()
        #print("label shape", label.shape)

        #Visualize and plot onto a png file
        #self.save_individual_heatmaps(label)
        #breakpoint()
        
        #self.save_aggregated_heatmap(label) #to be used during visualization
        #breakpoint()
        return video, label, label_mask

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
                video_augmentation.Resize(self.image_scale), #maybe yes
                video_augmentation.ToTensor(),
                #video_augmentation.TemporalRescale(0.2, self.frame_interval), #definately no
            ])
        else:
            print("Apply testing transform.")
            return video_augmentation.Compose([
                video_augmentation.CenterCrop(self.input_size),
                video_augmentation.Resize(self.image_scale),
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
        pose_output = [torch.tensor(pose, dtype=torch.float32) for pose in pose_output]
        pose_weights = [torch.tensor(w, dtype=torch.float32) for w in pose_weights]
        pose_weights = torch.stack(pose_weights, dim=0)
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
            video_length = torch.LongTensor([np.ceil(len(vid) / total_stride) * total_stride + 2*left_pad for vid in video])
            right_pad = int(np.ceil(max_len / total_stride)) * total_stride - max_len + left_pad
            max_len = max_len + left_pad + right_pad
            #print("left pad:", left_pad)
            #print("right pad:", right_pad)
            #Pad video
            padded_video = [torch.cat(
                (
                    vid[0][None].expand(left_pad, -1, -1, -1),
                    vid,
                    vid[-1][None].expand(max_len - len(vid) - left_pad, -1, -1, -1),
                )
                , dim=0)
                for vid in video]
            padded_video = torch.stack(padded_video)
            
            #Pad pose data
            padded_pose = pose_output
            '''
            padded_pose = [torch.cat(
                (
                pose[0][None].expand(left_pad, -1, -1),
                pose,
                pose[-1][None].expand(max_len - len(pose) - left_pad, -1, -1),
                ),dim=0)
                for pose in pose_output
            ]
            padded_pose = torch.stack(padded_pose)
            '''
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
        #Padding for pose data. 
        '''GloFE padding for pose data looks like this
            if X.shape[-1] == 2:
                padding = np.zeros((X.shape[0], X.shape[1], 1))
                X = np.concatenate((X, padding), axis=2)
        '''
        #Instead of pose, use heatmap data from GenerateTarget
        pose_length = video_length

        padded_video = torch.stack([
            torch.cat((vid[0:1], vid[0:1].expand(vid.shape[0] - 1, *vid.shape[1:])), dim=0)
            for vid in padded_video
        ])
        '''
        padded_pose = torch.stack([
            torch.cat((pose[0:1], pose[0:1].expand(pose.shape[0] - 1, *pose.shape[1:])), dim=0)
            for pose in padded_pose
        ])
        '''
        #print("Padded video shape:", padded_video.shape)
        return padded_video, video_length, padded_pose, pose_length, pose_weights

    def __len__(self):
        return len(self.inputs_list) - 1

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
