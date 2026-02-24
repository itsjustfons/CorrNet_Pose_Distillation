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

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class NormLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        outputs = torch.matmul(x, F.normalize(self.weight, dim=0))
        return outputs

class LearnableUpsampleConv(nn.Module):
    def __init__(self, in_channels=512, mid_channels=512):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=4, stride=2, padding=1)  # 7→14
        self.up2 = nn.ConvTranspose2d(mid_channels, mid_channels, kernel_size=4, stride=2, padding=1)  # 14→28
        self.up3 = nn.ConvTranspose2d(mid_channels, mid_channels, kernel_size=4, stride=2, padding=1)  # 28→56
        self.up4 = nn.ConvTranspose2d(mid_channels, mid_channels, kernel_size=4, stride=2, padding=1)  # 56→112
        self.up5 = nn.ConvTranspose2d(mid_channels, mid_channels, kernel_size=4, stride=2, padding=1) # 112→224
        #self.conv = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.up1(x))
        x = F.relu(self.up2(x))
        x = F.relu(self.up3(x))
        x = F.relu(self.up4(x))
        #x = F.relu(self.up5(x))
        #x = self.conv(x)
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return x


#Custom Spatial Weighting Loss Function
class SpatiallyWeightedMSELoss(nn.Module):
    def __init__(self, alpha=5.0, use_target_weight=True):
        super().__init__()
        self.alpha = alpha
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight=None):
        # output: [B, K, H, W]
        # target: [B, K, H, W]
        # target_weight: [B, K, 1]
        #print("target type", (type(target)))
        weight_map = 1.0 + self.alpha * target  # [B, K, H, W]

        #if shape of output, target and target weight is 5, flatten time and batch
        if len(output.shape) == 5:
            B, T, K, H, W = output.shape
            output = output.reshape(B*T, K, H, W)
            target = target.reshape(B*T, K, H, W)
            weight_map = weight_map.reshape(B*T, K, H, W)

        mse = (output - target) ** 2  # [B, K, H, W]. Apply squared error
        weighted_loss = mse * weight_map  # [B, K, H, W]. Apply weight map, and alpha weight

        # Mean over spatial dimensions
        per_joint_loss = weighted_loss.view(output.size(0), output.size(1), -1).mean(dim=2)  # [B, K]. Get mean

        # Mean over spatial and temporal dimensions
        #per_joint_loss = weighted_loss.view(output.size(0), output.size(2), -1).mean(dim=2)  # [B, K]
        #breakpoint()
        if self.use_target_weight and target_weight is not None:
            if len(target_weight.shape) == 4:
                B, T, K, I = target_weight.shape
                target_weight = target_weight.reshape(B*T, K, I)
            per_joint_loss = per_joint_loss * target_weight.squeeze(-1)  # [B, K]
        

        return per_joint_loss.mean()


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
        
        self.conv1d = TemporalConv(input_size=25088, #originially 512, but changed to match flattened input
                                   hidden_size=1024,
                                   conv_type=conv_type,
                                   use_bn=use_bn,
                                   num_classes=num_classes)
        
        
        #self.decoder = utils.Decode(gloss_dict, num_classes, 'beam')
        self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                          num_layers=2, bidirectional=True)
        
        #self.resolution_up = UpsampleConv(out_w=96, out_h=72)
        #self.resolution_up = LearnableUpsampleConv()

        #replace self.classifier with heatmap head
        #My custom keypoint head
        #set keypoint head config
        
        keypoint_head_topdown_heatmap = dict(
            type = 'TopdownHeatmapSimpleHead',
            in_channels=512,
            out_channels=133,
            num_deconv_layers=0,
            extra = dict(final_conv_kernel=1),
            loss_keypoint = dict(
                type = 'JointsMSELoss',
                use_target_weight=True
                )
            )
        
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
        
        #Keypoint head used in GloFE Extraction
        '''
        keypoint_head=dict(
            type='TopdownHeatmapSimpleHead',
            in_channels=48,
            out_channels=channel_cfg['num_output_channels'], #num_output_channels = 133 for 133 keypoints
            num_deconv_layers=0,
            extra=dict(final_conv_kernel=1, ),
            loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True))
        '''
            
        #initialize keypoint head with the set config
        self.keypoint_head = builder.build_head(keypoint_head_regression)

        #initialize MSEloss
        #self.heatmaploss = JointsMSELoss(use_target_weight=True)

        #heatmap loss coming straight from the head

        #using custom weighted mse loss
        #self.heatmaploss = SpatiallyWeightedMSELoss(alpha = 60.0)
        
        '''
        if weight_norm:
            self.classifier = NormLinear(hidden_size, self.num_classes)
            self.conv1d.fc = NormLinear(hidden_size, self.num_classes)
        else:
            self.classifier = nn.Linear(hidden_size, self.num_classes)
            self.conv1d.fc = nn.Linear(hidden_size, self.num_classes)
        if share_classifier:
            self.conv1d.fc = self.classifier
        '''
        #self.register_backward_hook(self.backward_hook)

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
            framewise = self.conv2d(x.permute(0,2,1,3,4))#.view(batch, temp, -1).permute(0,2,1) # btc -> bct. batch time, 512, 7, 7
            
            #flatten framewise height width channel to get (batch, time, C = 25088)
            B, T, C, H, W = framewise.shape
            framewise = framewise.view(B, T, C * H * W)
            framewise = framewise.permute(0,2,1) #batch channel time
            #breakpoint()
            #feed to conv1d
            conv1d_outputs = self.conv1d(framewise)
            framewise = conv1d_outputs['visual_feat']
            #breakpoint()
            #Issue here with contiguous
            framewise = framewise.permute(1, 0, 2)
            #framewise = framewise.permute(1, 0, 2).contiguous().view(B * T, C)

            B, T, C, = framewise.shape
            framewise = framewise.reshape(B * T, C) 

            #getting stats of framewise before keypoint head
            '''
            print("framewise mean:", framewise.mean())
            print("framewise max:", framewise.max())
            print("framewise min:", framewise.min())
            '''
            output = self.keypoint_head(framewise) 
            output = output.view(B, T, *output.shape[1:]) 
            #breakpoint()
        else:
            # frame-wise features
            framewise = x

        return {"predict_heatmap": output}

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
    
    def keypoint_loss(self, ret_dict, label, target_weight):
        target_weight = target_weight.expand(-1, -1, -1, 2)
        B, T, J, _ = target_weight.shape 
        #breakpoint()
        target_weight = target_weight.contiguous().view(B * T, J, 2)
        ret_dict = ret_dict.contiguous().view(B * T, J, 2)
        label = label[..., :2].contiguous().view(B * T, J, 2)


        #breakpoint()
        print("output max:",ret_dict[0].max())
        print("label max:", label[0].max())
        return self.keypoint_head.get_loss(
            output=ret_dict,
            target = label,
            target_weight = target_weight
            )

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
