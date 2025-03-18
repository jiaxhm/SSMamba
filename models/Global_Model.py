from models.vmamba import Backbone_VSSM
from torch import nn
import torch
from models.activations_autofn import LishAuto

__all__ = [
    'GEncoder1'
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

class GEncoder(nn.Module):
    def __init__(self, Dulbrn=24,
                 ckpt="/root/autodl-tmp/mamba_s/models/vssm_small_0229_ckpt_epoch_222.pth"):
        super(GEncoder, self).__init__()

        self.encoder = Backbone_VSSM(
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
        self.encoder.load_state_dict(checkpoint['model'], strict=False)

        # self.load_pretrained(ckpt)
        # self.encoder.eval()
        # for k, v in self.encoder.named_parameters():
        #     v.requires_grad = False
        self.up_concat4 = unetUp(384, 192, 384, 192)
        self.up_concat3 = unetUp(192, 96, 192, 96)
        self.up_concat2 = unetUp(96, 48, 96, 48)
        self.up_concat1 = unetUp(48, 24, 48, 24)

        self.sup = nn.Conv2d(24, 9, 1)
        self.softmax = nn.Softmax(1)
        self.edge = nn.Conv2d(24, 1, 1)

    def forward(self, x):

        global_feat = self.encoder(x)
        up4 = self.up_concat4(global_feat[3], global_feat[4])
        up3 = self.up_concat3(global_feat[2], up4)
        up2 = self.up_concat2(global_feat[1], up3)
        up1 = self.up_concat1(global_feat[0], up2)

        global_feature = [up1, up2, up3, up4, global_feat[4]]

        final = self.sup(up1)
        sup = self.softmax(final)
        edge = self.edge(up1)

        if self.training:
            return global_feature#sup, edge#global_feature#sup, edge, global_feature    #global_feature#sup, edge
        else:
            return global_feature#sup#global_feature#sup, edge, global_feature    #global_feature#sup

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

def GEncoder1(data=None):
    # Model without  batch normalization
    model = GEncoder()
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model

if __name__ == '__main__':
    x = torch.randn(2, 3, 208, 208).cuda()
    b, c, h, w = x.shape
    net = GEncoder().cuda()
    net.eval()
    y = net(x)