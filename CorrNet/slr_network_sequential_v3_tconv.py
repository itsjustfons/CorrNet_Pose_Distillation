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
import modules.resnet_seq as resnet
import os
import matplotlib.pyplot as plt
#import mm
from mmpose.models.heads import topdown_heatmap_base_head
from mmpose.models import builder
from mmpose.models.losses.mse_loss import JointsMSELoss

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
                                   conv_type=9,
                                   use_bn=use_bn,
                                   num_classes=num_classes)
        
        
        #self.decoder = utils.Decode(gloss_dict, num_classes, 'beam')
        self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                          num_layers=2, bidirectional=True)
        
        
        #losses for reference
        mseloss = dict(type = 'MSELoss',use_target_weight=True,)
        mpjpeloss = dict(type = 'MPJPELoss', use_target_weight=True)

        keypoint_head_regression = dict(
            type = 'DeepposeRegressionHead',
            in_channels=1024,
            num_joints=133,
            out_sigma = False,
            loss_keypoint = mpjpeloss
            )
            
        #initialize keypoint head with the set config
        self.keypoint_head = builder.build_head(keypoint_head_regression)

        #initialize temporal keypoint head
        self.temporal_keypoint_head = nn.Linear(1024, 266)
        
        #initialize confidence head
        self.conf_head = nn.Linear(1024, 133) 

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
            batch, temp, channel, height, width = x.shape
            framewise = self.conv2d(x.permute(0,2,1,3,4)).view(batch, temp, -1).permute(0,2,1) # btc -> bct. batch time, 512, 7, 7
            #B, T, C, H, W = framewise.shape
            #framewise = framewise.view(B, T, C * H * W)
            #pass through 1d convolution
            #framewise = framewise.permute(0,2,1) #batch channel time
            breakpoint()
            framewise = self.conv1d(framewise, len_x)['visual_feat'].permute(1,0,2)

            #framewise = framewise.view(B,T, C * H * W)

            #get xy coordinates
            output_xy = self.temporal_keypoint_head(framewise)
            output_xy = output_xy.view(batch, temp, 133, 2) 
            output_xy = torch.relu(output_xy)

            #get confidence values
            output_conf = self.conf_head(framewise).unsqueeze(-1)
            output_conf = torch.sigmoid(output_conf)

            output = torch.cat([output_xy, output_conf], dim=-1)
            #output = output.view(B, T, *output.shape[1:]) #return to original BTC shape
        else:
            # frame-wise features
            framewise = x

        # Feed framewise into pose head. Skip conv1d and BiLSTM. Insert MMPose
        
        ''' #turn off conv1d and BILSTM for now
        conv1d_outputs = self.conv1d(framewise, len_x)
        # x: T, B, C
        x = conv1d_outputs['visual_feat']
        lgt = conv1d_outputs['feat_len']
        tm_outputs = self.temporal_model(x, lgt)
        outputs = self.classifier(tm_outputs['predictions'])
        
        ## Put in a decoding part like this for decoding heatmaps into keypoints for model evaluation ##
        pred = None if self.training \
            else self.decoder.decode(outputs, lgt, batch_first=False, probs=False)
        conv_pred = None if self.training \
            else self.decoder.decode(conv1d_outputs['conv_logits'], lgt, batch_first=False, probs=False)
        breakpoint()
        ''' 
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
