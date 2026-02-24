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
#import mm
from mmpose.models.heads import topdown_heatmap_base_head
from mmpose.models import builder
from mmpose.models.losses.mse_loss import JointsMSELoss
#final linear layers have these additional functions added at the end:
#XY head: ReLU
#Confidence: Sigmoid

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

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
        
        self.conv1d = TemporalConv(input_size=512,
                                   hidden_size=hidden_size,
                                   conv_type=conv_type,
                                   use_bn=use_bn,
                                   num_classes=num_classes)
        
        
        #self.decoder = utils.Decode(gloss_dict, num_classes, 'beam')
        self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                          num_layers=2, bidirectional=True)
        
        
        #losses for reference
        mpjpeloss = dict(type = 'MPJPELoss', use_target_weight=True)

        keypoint_head_regression = dict(
            type = 'DeepposeRegressionHead',
            in_channels=25088,
            num_joints=133,
            out_confidence = False,
            out_sigma = False,

            loss_keypoint = mpjpeloss
            )
        self.keypoint_head = builder.build_head(keypoint_head_regression)
        
        # Confidence head: simple linear projection
        self.conf_head = nn.Linear(25088, 133)  # one confidence per joint

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

    def forward(self, x, len_x=None, label=None, label_lgt=None):
        if len(x.shape) == 5:
            #breakpoint()
            framewise = self.conv2d(x.permute(0,2,1,3,4).contiguous())#.view(batch, temp, -1).permute(0,2,1) # btc -> bct. batch time, 512, 7, 7
            B, T, C, H, W = framewise.shape
            framewise = framewise.view(B * T, C, H, W) 
            framewise = framewise.view(framewise.size(0), -1) #Flatten C, H, W into 25088 channel vector
            
            output_xy = self.keypoint_head(framewise) 
            output_xy = torch.relu(output_xy)
            output_xy = output_xy.view(B, T, *output_xy.shape[1:]) 
            
            output_conf = self.conf_head(framewise)
            output_conf = torch.sigmoid(output_conf)
            output_conf = output_conf.view(B, T, 133, 1)

            output = torch.cat([output_xy, output_conf], dim=-1)
        #breakpoint()
        return {
            "predict_heatmap": output
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
    
    def save_aggregated_heatmap_first_sample(self, output, output_path='/data/group1/z40575r/CorrNet_pose_distillation/training_heatmaps_aggregated.png'):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        output = torch.tensor(output)  # Ensure it's a tensor

        aggregated = output[0].sum(dim=0)  # Sum over 133 keypoints → shape (72, 96)
        aggregated = aggregated / aggregated.max()  # Normalize to [0, 1]

        plt.imshow(aggregated.cpu().numpy(), cmap='hot', interpolation='nearest')
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
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
