import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init

from ..builder import HEADS, build_loss


@HEADS.register_module()
class TianchiBaseHead(nn.Module):

    def __init__(self,
                 in_channels,
                 feat_channels=256,
                 num_points=11,
                 num_disc_classes=5,
                 num_vertebra_classes=2,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_mask=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None):
        super(TianchiBaseHead, self).__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.num_points = num_points
        self.num_disc_classes = num_disc_classes
        self.num_vertebra_classes = num_vertebra_classes
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.loss_mask = build_loss(loss_mask)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self._init_layers()

    def _init_layers(self):
        self.point_convs = nn.ModuleList()
        # self.cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.point_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            # self.cls_convs.append(
            #     ConvModule(
            #         chn,
            #         self.feat_channels,
            #         3,
            #         stride=1,
            #         padding=1,
            #         conv_cfg=self.conv_cfg,
            #         norm_cfg=self.norm_cfg))
        self.point_pred = nn.Conv2d(
            self.feat_channels, self.num_points, 3, padding=1)
        # self.disc_cls = nn.Conv2d(
        #     self.feat_channels, self.num_disc_classes, 3, padding=1)
        # self.vertebra_cls = nn.Conv2d(
        #     self.feat_channels, self.num_vertebra_classes, 3, padding=1)

    def init_weights(self):
        for m in self.point_convs:
            normal_init(m.conv, std=0.01)
        # for m in self.cls_convs:
        #     normal_init(m.conv, std=0.01)
        # bias_cls = bias_init_with_prob(0.01)
        normal_init(self.point_pred, std=0.01)
        # normal_init(self.disc_cls, std=0.01, bias=bias_cls)
        # normal_init(self.vertebra_cls, std=0.01, bias=bias_cls)

    def forward(self, x):
        # cls_feat = x
        point_feat = x
        # for cls_conv in self.cls_convs:
        #     cls_feat = cls_conv(cls_feat)
        for point_conv in self.point_convs:
            point_feat = point_conv(point_feat)

        point_pred = self.point_pred(point_feat)
        # disc_score = self.disc_cls(cls_feat)
        # vertebra_score = self.vertebra_cls(cls_feat)

        # return disc_score, vertebra_score, point_pred
        return point_pred

    def forward_train(self, x, img_metas, gt_points, gt_labels, valid):
        # disc_score, vertebra_score, point_pred = self(x)
        point_pred = self(x)

        device = point_pred.device
        mask_list = []
        weight_list = []
        for img_meta, _gt_points, _gt_labels, _valid in zip(
                img_metas, gt_points, gt_labels, valid):
            masks, weights = self.get_masks(
                img_meta, _gt_points, _gt_labels, _valid, device=device)
            mask_list.append(masks)
            weight_list.append(weights)
        mask = mask_list[0] if len(mask_list) == 0 else torch.cat(
            mask_list, dim=0)
        weight = weight_list[0] if len(weight_list) == 0 else torch.cat(
            weight_list, dim=0)

        label = mask.reshape(-1)
        label_weight = weight.reshape(-1)
        point_pred = point_pred.reshape(-1).unsqueeze(1)

        # num_total_samples = len(img_metas) * self.num_points
        loss_mask = self.loss_mask(point_pred, label, label_weight)

        return dict(loss_mask=loss_mask)

    def get_masks(self, img_meta, gt_points, gt_labels, valid, device):
        num_frames = len(img_meta['img_shapes'])

        img_shapes = img_meta[
            'pad_shapes'] if 'pad_shapes' in img_meta else img_meta[
                'img_shapea']
        height, width = img_shapes[0]
        masks = torch.zeros((num_frames, self.num_points, height, width),
                            dtype=torch.int64,
                            device=device)
        weights = torch.zeros_like(masks)

        gt_idx = img_meta['gt_idx']
        for i in range(gt_points.shape[0]):
            point = gt_points[i].to(torch.int64)
            masks[gt_idx, i, point[1], point[0]] = 1
            weights[gt_idx, i, ...] = valid[i]

        return masks, weights
