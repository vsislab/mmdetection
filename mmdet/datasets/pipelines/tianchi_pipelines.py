import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC

from .formating import to_tensor
from ..builder import PIPELINES


@PIPELINES.register_module()
class TianchiLoadAnnotation(object):

    def __init__(self,
                 with_valid=False,
                 with_points=True,
                 with_bboxes=False,
                 with_labels=True):
        self.with_valid = with_valid
        self.with_points = with_points
        self.with_bboxes = with_bboxes
        self.with_labels = with_labels

    def __call__(self, results):
        ann_info = results['ann_info']
        if self.with_valid:
            results['valid'] = ann_info['valid'].copy()
        if self.with_points:
            results['gt_points'] = ann_info['points'].copy()
        if self.with_bboxes:
            results['gt_bboxes'] = ann_info['bboxes'].copy()
        if self.with_labels:
            results['gt_labels'] = ann_info['labels'].copy()

        return results


@PIPELINES.register_module()
class TianchiFormatBundle(object):

    def __call__(self, results):
        img = results['img']
        if len(img.shape) < 3:
            img = np.expand_dims(img, -1)
        img = np.ascontiguousarray(img.transpose(2, 0, 1))
        results['img'] = DC(to_tensor(img), stack=True)
        for key in ['gt_points', 'gt_labels', 'valid']:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]))

        return results


@PIPELINES.register_module()
class TianchiResize(object):

    def __init__(self, img_scale=None, keep_ratio=True):
        self.img_scale = img_scale
        self.keep_ratio = keep_ratio

    def _resize_imgs(self, results):
        img = results['img']

        if self.keep_ratio:
            img, scale_factor = mmcv.imrescale(
                img, self.img_scale, return_scale=True)
            new_h, new_w = img.shape[:2]
            h, w = results['img'].shape[:2]
            w_scale = new_w / w
            h_scale = new_h / h
        else:
            img, w_scale, h_scale = mmcv.imresize(
                img, self.img_scale, return_scale=True)

        results['img'] = img
        scale_factor = np.array([w_scale, h_scale], dtype=np.float32)

        results['img_shape'] = img.shape
        # in case that there is no padding
        results['pad_shape'] = img.shape
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio

    def _resize_points(self, results):
        """Resize points with ``results['scale_factor']``."""
        img_shape = results['img_shape']
        points = results['gt_points'] * results['scale_factor']
        points[:, 0] = np.clip(points[:, 0], 0, img_shape[1])
        points[:, 1] = np.clip(points[:, 1], 0, img_shape[0])

        results['gt_points'] = points

    def __call__(self, results):
        self._resize_imgs(results)
        self._resize_points(results)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale},'
        repr_str += f' keep_ratio={self.keep_ratio})'

        return repr_str


@PIPELINES.register_module()
class TianchiPad(object):

    def __init__(self, size=None, pad_val=0):
        self.size = size
        self.pad_val = pad_val

    def _pad_img(self, results):
        if self.size is not None:
            padded_img = mmcv.impad(
                results['img'], shape=self.size, pad_val=self.pad_val)

            results['img'] = padded_img
            results['pad_shape'] = padded_img.shape

    def __call__(self, resutls):
        self._pad_img(resutls)

        return resutls
