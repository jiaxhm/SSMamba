from models.vmamba import Backbone_VSSM
from models.Global_Model import GEncoder
from torch import nn
import torch
import math
import random
from torch.nn.functional import interpolate
from torch.nn import functional as F
from models.activations_autofn import LishAuto

__all__ = [
    'LEncoder1'
]

act = LishAuto(inplace=True)

class unetUp(nn.Module):
    def __init__(self, in_size, out_size, upsize, mid_size):  #channel or spatial
        super(unetUp, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(upsize, mid_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(mid_size),
            act
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_size),
            act
        )

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        return outputs

class SFTLayer(nn.Module):
    def __init__(self, head_channels):
        super(SFTLayer, self).__init__()

        self.SFT_scale_conv0 = nn.Conv2d(head_channels, head_channels, 1)
        self.SFT_scale_conv1 = nn.Conv2d(head_channels, head_channels, 1)

        self.SFT_shift_conv0 = nn.Conv2d(head_channels, head_channels, 1)
        self.SFT_shift_conv1 = nn.Conv2d(head_channels, head_channels, 1)

    def forward(self, local_features, global_features):
        scale = self.SFT_scale_conv1(act(self.SFT_scale_conv0(global_features)))
        shift = self.SFT_shift_conv1(act(self.SFT_shift_conv0(global_features)))
        fuse_features = local_features * (scale + 1) + shift
        return fuse_features
    

class SFTLayerl(nn.Module):
    def __init__(self, head_channels):
        super(SFTLayerl, self).__init__()

        self.SFT_scale_conv0 = nn.Conv2d(head_channels, head_channels, 1)
        self.SFT_scale_conv1 = nn.Conv2d(head_channels, head_channels, 1)

        self.SFT_shift_conv0 = nn.Conv2d(head_channels, head_channels, 1)
        self.SFT_shift_conv1 = nn.Conv2d(head_channels, head_channels, 1)

    def forward(self, local_features, global_features):
        scale = self.SFT_scale_conv1(act(self.SFT_scale_conv0(local_features)))
        shift = self.SFT_shift_conv1(act(self.SFT_shift_conv0(local_features)))
        fuse_features = global_features * (scale + 1) + shift
        return fuse_features


class Local8x8_fuse_head(nn.Module):
    def __init__(self,
                 mla_channels=128,
                 mlahead_channels=128,
                 norm_layer=nn.BatchNorm2d,
                 activate=nn.Identity):
        super(Local8x8_fuse_head, self).__init__()

        self.channels = mla_channels
        self.head_channels = mlahead_channels
        self.BatchNorm = norm_layer
        self.activate = activate()

        self.SFT_head = SFTLayer(self.head_channels)
        self.SFT_head1 = SFTLayerl(self.head_channels)
        self.edge_head = nn.Sequential(
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.head_channels), act,
            nn.Conv2d(self.head_channels, 48, 3, padding=1, bias=False),
            nn.BatchNorm2d(48), act,
            nn.Conv2d(48, 1, 1)
        )

        self.superpixel_head = nn.Sequential(
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.head_channels), act,
            nn.Conv2d(self.head_channels, 48, 3, padding=1, bias=False),
            nn.BatchNorm2d(48), act,
            nn.Conv2d(48, 9, 1),
            nn.Softmax(dim=1)
        )
        self.global_rate = torch.nn.Parameter(torch.Tensor([1.0]))
        self.local_rate = torch.nn.Parameter(torch.Tensor([1.0]))

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight.data)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def forward(self, local_features, global_features):

        fuse_features = self.SFT_head(local_features, global_features)
        fuse_features1 = self.SFT_head1(local_features, global_features)
        fuse_feature = self.global_rate * fuse_features + self.local_rate + fuse_features1
        fuse_edge = self.edge_head(fuse_feature)
        fuse_superpixel = self.superpixel_head(fuse_feature)
        return fuse_superpixel, fuse_edge

def cat_patch(f00i, f01i, f10i, f11i, fg):

    cat_f0 = torch.cat([f00i, f01i], dim=3)
    cat_f1 = torch.cat([f10i, f11i], dim=3)
    cat_combined = torch.cat([cat_f0, cat_f1], dim=2)
    interpolated = interpolate(cat_combined,size=fg.size()[2:],  mode="bilinear")

    return interpolated

class LEncoder(nn.Module):
    def __init__(self, Dulbrn=24,
                 ckpt="/root/autodl-tmp/mamba_s/models/vssm_small_0229_ckpt_epoch_222.pth"):
        super(LEncoder, self).__init__()

        self.local_model = Backbone_VSSM(
            pretrained=None,
            out_indices=(0, 1, 2),
            # out_indices=(0, 1, 2, 3),
            dims=96,
            # depths=(2, 2, 15, 2),
            depths=(2, 2, 15, 0),
            ssm_d_state=1,
            ssm_dt_rank="auto",
            ssm_ratio=2.0,
            ssm_conv=3,
            ssm_conv_bias=False,
            forward_type="v05_noz",  # v3_noz,
            mlp_ratio=4.0,
            downsample_version="v3",
            patchembed_version="v2",
            drop_path_rate=0.3,
            Dulbrn=Dulbrn)
        checkpoint = torch.load(ckpt, map_location='cpu')
        self.local_model.load_state_dict(checkpoint['model'], strict=False)

        self.global_encoder = GEncoder(Dulbrn=24).cuda()
        global_ckpt = '/root/autodl-tmp/mamba_s/CKPT/VMamba_LELU/BSD500/GEncoder1_adam_3000000epochs_epochSize6000_b16_lr0.0005_posW0.003_24_11_13_15_44/model_best.tar'###
        checkpoint = torch.load(global_ckpt, map_location='cpu')
        self.global_encoder.load_state_dict(checkpoint['state_dict'])#, strict=False
        self.global_encoder.eval()
        print(self.global_encoder.training)
        for k, v in self.global_encoder.named_parameters():
            v.requires_grad = False

        self.fuse_head = Local8x8_fuse_head(24, 24)
        self.up_concat4 = unetUp(384, 192, 384, 192)
        self.up_concat3 = unetUp(192, 96, 192, 96)
        self.up_concat2 = unetUp(96, 48, 96, 48)
        self.up_concat1 = unetUp(48, 24, 48, 24)

    def forward(self, x):

        global_feature = self.global_encoder(x)

        x = interpolate(x, scale_factor=2, mode="bilinear")
        _, _, H, W = x.size()

        grad = random.randint(0, 4)
        if grad == 0:
            feat_00 = self.local_model(x[..., :H // 2 + 1, :W // 2 + 1])
            with torch.no_grad():
                feat_01 = self.local_model(x[..., :H // 2 + 1, W // 2 + 1:])
                feat_10 = self.local_model(x[..., H // 2 + 1:, :W // 2 + 1])
                feat_11 = self.local_model(x[..., H // 2 + 1:, W // 2 + 1:])
        elif grad == 1:
            feat_01 = self.local_model(x[..., :H // 2 + 1, W // 2 + 1:])
            with torch.no_grad():
                feat_00 = self.local_model(x[..., :H // 2 + 1, :W // 2 + 1])
                feat_10 = self.local_model(x[..., H // 2 + 1:, :W // 2 + 1])
                feat_11 = self.local_model(x[..., H // 2 + 1:, W // 2 + 1:])
        elif grad == 2:
            feat_10 = self.local_model(x[..., H // 2 + 1:, :W // 2 + 1])
            with torch.no_grad():
                feat_00 = self.local_model(x[..., :H // 2 + 1, :W // 2 + 1])
                feat_01 = self.local_model(x[..., :H // 2 + 1, W // 2 + 1:])
                feat_11 = self.local_model(x[..., H // 2 + 1:, W // 2 + 1:])
        else:
            feat_11 = self.local_model(x[..., H // 2 + 1:, W // 2 + 1:])
            with torch.no_grad():
                feat_00 = self.local_model(x[..., :H // 2 + 1, :W // 2 + 1])
                feat_01 = self.local_model(x[..., :H // 2 + 1, W // 2 + 1:])
                feat_10 = self.local_model(x[..., H // 2 + 1:, :W // 2 + 1])
        
        local_feat = [self.cat_patch(f00i, f01i, f10i, f11i, fg)
                      for f00i, f01i, f10i, f11i, fg in
                      zip(feat_00, feat_01, feat_10, feat_11, global_feature)]
        
        # print(f"feat_00 length: {len(feat_00)}")
        # print(f"feat_01 length: {len(feat_01)}")
        # print(f"feat_10 length: {len(feat_10)}")
        # print(f"feat_11 length: {len(feat_11)}")
        # print(f"global_feature length: {len(global_feature)}")
        # print(global_feature[0].shape)
        # print(global_feature[1].shape)
        
        # local_feat0 = cat_patch(feat_00[0], feat_01[0], feat_10[0], feat_11[0], global_feature[0])
        # local_feat1 = cat_patch(feat_00[1], feat_01[1], feat_10[1], feat_11[1], global_feature[1])
        # local_feat2 = cat_patch(feat_00[2], feat_01[2], feat_10[2], feat_11[2], global_feature[2])
        # local_feat3 = cat_patch(feat_00[3], feat_01[3], feat_10[3], feat_11[3], global_feature[3])
        # local_feat4 = cat_patch(feat_00[4], feat_01[4], feat_10[4], feat_11[4], global_feature[4])
        # local_feat = [local_feat0, local_feat1, local_feat2, local_feat3, local_feat4]
        
        # print(f"local_feat length: {len(local_feat)}")
        
        up4 = self.up_concat4(local_feat[3], local_feat[4])
        up3 = self.up_concat3(local_feat[2], up4)
        up2 = self.up_concat2(local_feat[1], up3)
        up1 = self.up_concat1(local_feat[0], up2)

        local_feature = [up1, up2, up3, up4, local_feat[4]]

        sup, edge = self.fuse_head(local_feature[0], global_feature[0])

        if self.training:
            return sup, edge
        else:
            return sup
        
        
    def cat_patch(self, f00i, f01i, f10i, f11i, fg):
        return interpolate(
            torch.cat([torch.cat([f00i, f01i], dim=3), torch.cat([f10i, f11i], dim=3)], dim=2),
            size=fg.size()[2:],
            mode="bilinear")
    
        
        
        
    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

def LEncoder1(data=None):
    # Model without  batch normalization
    model = LEncoder()
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model

    
    


if __name__ == '__main__':
    x = torch.randn(2, 3, 208, 208).cuda()
    b, c, h, w = x.shape
    net = LEcoder().cuda()
    y = net(x)