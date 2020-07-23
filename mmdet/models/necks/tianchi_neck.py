import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
from mmcv.cnn.bricks import NonLocal2d

from ..builder import NECKS


@NECKS.register_module()
class TianchiTopBottom(nn.Module):

    def __init__(self,
                 in_channels,
                 num_levels,
                 out_size=(512, 512),
                 refine_type=None,
                 conv_cfg=None,
                 norm_cfg=None):
        super(TianchiTopBottom, self).__init__()
        assert refine_type in [None, 'conv', 'non_local']

        self.in_channels = in_channels
        self.num_levels = num_levels
        self.out_size = out_size
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.refine_type = refine_type
        if self.refine_type == 'conv':
            self.refine = ConvModule(
                self.in_channels,
                self.in_channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)
        elif self.refine_type == 'non_local':
            self.refine = NonLocal2d(
                self.in_channels,
                reduction=1,
                use_scale=False,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)

    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == self.num_levels

        feats = []
        for i in range(self.num_levels):
            gathered = F.interpolate(
                inputs[i], size=self.out_size, mode='nearest')
            feats.append(gathered)

        feat = sum(feats) / len(feats)
        if self.refine_type is not None:
            feat = self.refine(feat)

        return feat
