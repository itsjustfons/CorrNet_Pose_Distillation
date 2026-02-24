import pdb
import copy
import utils
import torch
import types
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from modules.criterions import SeqKD
from modules import BiLSTMLayer, TemporalConv
import modules.resnet as resnet
import os
import matplotlib.pyplot as plt
from utils.beam_search import AutoRegressiveBeamSearch
from models.trans_model_inter_vn import TransBaseModel
from types import SimpleNamespace

#import mm
from mmpose.models.heads import topdown_heatmap_base_head
from mmpose.models import builder
from mmpose.models.losses.mse_loss import JointsMSELoss

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

#Default args parser for GloFE model
args = {
    # ======================
    # General
    # ======================
    "vocab_size": 25000,
    "dim_embedding": 768,
    "activation": "gelu",
    "norm_first": False,
    "mask_future": True,
    "froze_vb": False,

    # ======================
    # Encoder configs
    # ======================
    "num_enc": 4,
    "dim_forward_enc": 1024,
    "nhead_enc": 8,
    "dropout_enc": 0.1,
    "pe_enc": True,
    "mask_enc": True,

    # ======================
    # Decoder configs
    # ======================
    "num_dec": 4,
    "dim_forward_dec": 1024,
    "nhead_dec": 8,
    "dropout_dec": 0.1,

    # ======================
    # Loss configs
    # ======================
    "ls": 0.2,

    "inter_cl": True,
    "inter_cl_margin": 0.4,
    "inter_cl_alpha": 1.0,

    # Embedding configuration for loss
    "inter_cl_vocab": 5523,
    "inter_cl_we_dim": 300,
    "inter_cl_we_path": "/data/group1/z40575r/CorrNet_pose_distillation/CorrNet/notebooks/openasl-v1.0/uncased_filtred_glove_VN_embed.pkl",

    # ======================
    # Backbone config
    # ======================
    "pose_backbone": "PartedPoseBackbone",

    # ======================
    # Training / runtime
    # ======================
    "work_dir_prefix": "/mnt/workspace/slt_baseline/work_dir",
    "work_dir": "checkpoints",
    "prefix": "phoenix_prefix",

    "epochs": 90,
    "bs": 16,
    "warm_up": 1000,
    "lr": 0.0003,
    "save_every": 5,

    "phase": "test",        # ['train', 'test']
    "split": "dev",           # ['train', 'dev', 'val', 'test', 'valid']
    "weights": None,
    "resume": -1,

    # ======================
    # DDP related
    # ======================
    "seed": 42,
    "ngpus": 1,
    "local_rank": 0,

    # ======================
    # Dataset configs
    # ======================
    "feat_path": None,   # path to OpenASL folder
    "label_path": None,  # path to OpenASL CSV labels
    "clip_length": 10,
    "tokenizer": "/mnt/workspace/slt_baseline/notebooks/openasl-bpe25000-tokenizer",
    "eos_token": "</s>",

    # ======================
    # Generator config
    # ======================
    "max_gen_tks": 35,
    "num_beams": 5,
}

args = SimpleNamespace(**args)

#To be used for constructing GloFE
def construct_model(model_cls, args, distributed=False):
    #rank = args.local_rank
    rank = 0
    #breakpoint()
    generator = AutoRegressiveBeamSearch(
        eos_index=2,
        max_steps=args.max_gen_tks,
        beam_size=args.num_beams,
        per_node_beam_size=2,
    )
    model = model_cls(args, generator=generator, sos_index=0)
    # move model to GPU
    device = torch.device(f'cuda:{rank}')
    model.to(device)
    '''
    if distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(
            model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=True)
    '''
    return model

class SLRModel(nn.Module):
    def __init__(
            self, num_classes, c2d_type, conv_type, use_bn=False,
            hidden_size=1024, gloss_dict=None, loss_weights=None,
            weight_norm=True, share_classifier=True
    ):
        super(SLRModel, self).__init__()
        self.decoder = None
        self.loss = dict()
        self.criterion_init()
        self.num_classes = num_classes
        self.loss_weights = loss_weights
        #self.conv2d = getattr(models, c2d_type)(pretrained=True)
        self.conv2d = getattr(resnet, c2d_type)()
        self.conv2d.fc = Identity()
        '''
        self.conv1d = TemporalConv(input_size=512,
                                   hidden_size=hidden_size,
                                   conv_type=conv_type,
                                   use_bn=use_bn,
                                   num_classes=num_classes)
        '''
        #self.decoder = utils.Decode(gloss_dict, num_classes, 'beam')
        
        #self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
        #                                  num_layers=2, bidirectional=True)
        
        
        #losses for reference
        mseloss = dict(type = 'MSELoss',use_target_weight=True,)
        mpjpeloss = dict(type = 'MPJPELoss', use_target_weight=True)

        keypoint_head_regression = dict(
            type = 'DeepposeRegressionHead',
            in_channels=25088,
            num_joints=133,
            out_sigma = False,
            loss_keypoint = mpjpeloss
            )
            
        #initialize keypoint head with the set config
        self.keypoint_head = builder.build_head(keypoint_head_regression)

        #initialize temporal keypoint head
        self.temporal_keypoint_head = nn.Linear(25088, 266)
        
        #initialize confidence head
        self.conf_head = nn.Linear(25088, 133) 

        model_cls = TransBaseModel
        #initialize GloFE and load weights
        self.glofe = construct_model(model_cls, args, False)
        glofe_weights_path = "/data/group1/z40575r/GloFE/work_dir/openasl/vm_model/glofe_vn_openasl.pt"
        glofe_weights = torch.load(glofe_weights_path)
        state = {k.replace("module.", "", 1): v for k, v in glofe_weights.items()}
        #breakpoint()
        self.glofe.load_state_dict(state)
        for p in self.glofe.parameters():
            p.requires_grad = False

        self.glofe.eval()

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0

    def masked_bn(self, inputs, len_x):
        def pad(tensor, length):
            return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

        x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])
        x = self.conv2d(x)
        x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
                       for idx, lgt in enumerate(len_x)])
        return x

    def forward(self, x, len_x=None, label=None, label_lgt=None, tokens=None, mask=None, token_length=None, vn_idxs=None, vn_len=None):
        if len(x.shape) == 5:
            framewise = self.conv2d(x.permute(0,2,1,3,4))#.view(batch, temp, -1).permute(0,2,1) # btc -> bct. batch time, 512, 7, 7
            B, T, C, H, W = framewise.shape
            framewise = framewise.view(B,T, C * H * W)

            #get xy coordinates
            output_xy = self.temporal_keypoint_head(framewise)
            output_xy = output_xy.view(B, T, 133, 2) 
            output_xy = torch.relu(output_xy)

            #get confidence values
            output_conf = self.conf_head(framewise).unsqueeze(-1)
            output_conf = torch.sigmoid(output_conf)

            keypoints = torch.cat([output_xy, output_conf], dim=-1)
            
            #feed keypoints into GloFE
            #breakpoint()
            output = self.glofe(
                x=keypoints,
                x_length=len_x,
                tgt=tokens,
                tgt_length=token_length,
                vn_idxs=vn_idxs,
                vn_len=vn_len
            )

        #breakpoint()
        return {
            "predict_heatmap": keypoints,
            "glofe_outputs": output
        }
    #For visualizing heatmaps
    def save_aggregated_heatmap(self, output, output_path='/data/group1/z40575r/CorrNet_pose_distillation/training_heatmaps_aggregated_dev.png'):
        # output: (B, 133, 72, 96)
        output = torch.tensor(output)  # Ensure tensor

        aggregated = output.sum(dim=(0, 1))  # Sum over batch and keypoints → shape (72, 96)
        aggregated = aggregated / aggregated.max()  # Normalize

        plt.imshow(aggregated.cpu().numpy(), cmap='hot', interpolation='nearest')
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    def save_aggregated_heatmaps_per_sample(self, output, output_dir='/data/group1/z40575r/CorrNet_pose_distillation/training_heatmaps_aggregated_dev.png'):
        os.makedirs(output_dir, exist_ok=True)
        output = torch.tensor(output)

        for i in range(output.shape[0]):  # Iterate over batch
            aggregated = output[i].sum(dim=0)  # shape (72, 96)
            aggregated = aggregated / aggregated.max()

            plt.imshow(aggregated.cpu().numpy(), cmap='hot', interpolation='nearest')
            plt.axis('off')
            plt.savefig(os.path.join(output_dir, f'sample_{i:02d}.png'), bbox_inches='tight', pad_inches=0)
            plt.close()

    def keypoint_loss(self, ret_dict, label, target_weight, alpha = 1.0, beta = 1.0, conf_mode = "MSE"):
        #target_weight = target_weight.expand(-1, -1, -1, 2)
        B, T, J, C = ret_dict.shape 
        if C == 2:
            return self.keypoint_head.get_loss(
                output=ret_dict,
                target = label,
                target_weight = target_weight
                )
        
        elif C == 3:
            student_xy = ret_dict[..., :2]
            teacher_xy = label[..., :2]

            student_conf = ret_dict[..., 2]
            teacher_conf = label[..., 2]

            #MPJPE on xy
            loss_xy = self.keypoint_head.get_loss(
                output = student_xy,
                target = teacher_xy,
                target_weight = target_weight
            )['reg_loss']

            #normalize loss
            loss_xy = loss_xy / 224

            #MSE on confidence value
            if conf_mode == "MSE":
                loss_conf = F.mse_loss(student_conf, teacher_conf)
            #L2 loss on confidence value
            elif conf_mode == "smooth_l1":
                loss_conf = F.smooth_l1_loss(student_conf, teacher_conf)
            #weighted sum
            loss = alpha * loss_xy + beta * loss_conf
            #breakpoint()
            return loss
        

    def criterion_calculation(self, ret_dict, label, label_lgt):
        loss = 0
        for k, weight in self.loss_weights.items():
            if k == 'ConvCTC':
                loss += weight * self.loss['CTCLoss'](ret_dict["conv_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
            elif k == 'SeqCTC':
                loss += weight * self.loss['CTCLoss'](ret_dict["sequence_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
            elif k == 'Dist':
                loss += weight * self.loss['distillation'](ret_dict["conv_logits"],
                                                           ret_dict["sequence_logits"].detach(),
                                                           use_blank=False)
        return loss

    def criterion_init(self):
        self.loss['CTCLoss'] = torch.nn.CTCLoss(reduction='none', zero_infinity=False)
        self.loss['distillation'] = SeqKD(T=8)
        return self.loss
