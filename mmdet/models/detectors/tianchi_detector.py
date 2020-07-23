import torch.nn as nn

from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector


@DETECTORS.register_module()
class TianchiDetector(BaseDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(TianchiDetector, self).__init__()
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        head.update(train_cfg=train_cfg)
        head.update(test_cfg=test_cfg)
        self.head = build_head(head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(TianchiDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.head.init_weights()

    def _process(self, img):
        img = img.view(-1, 1, img.size(3), img.size(4))

        return img

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        img = self._process(img)
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self, img, img_metas, gt_points, gt_labels, valid,
                      **kwargs):
        x = self.extract_feat(img)
        losses = self.head.forward_train(x, img_metas, gt_points, gt_labels,
                                         valid, **kwargs)

        return losses

    def simple_test(self, img, img_metas, rescale=False):
        raise NotImplementedError

    def aug_test(self, imgs, img_metas, rescales=False):
        raise NotImplementedError
