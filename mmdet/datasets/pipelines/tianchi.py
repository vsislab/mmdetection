import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC

from .formating import to_tensor
from ..builder import PIPELINES


@PIPELINES.register_module()
class TianchiLoadImageFromDICOM(object):

    def __call__(self, results):
        imgs = [dicom.image for dicom in results['dicom']]

        results['imgs'] = imgs
        results['ori_shapes'] = [img.shape for img in imgs]
        results['img_shapes'] = [img.shape for img in imgs]

        return results


@PIPELINES.register_module()
class TianchiLoadAnnotation(object):

    def __call__(self, results):
        ann_info = results['ann_info']
        results['valid'] = ann_info['valid'].copy()
        results['gt_points'] = ann_info['gt_points'].copy()
        results['gt_labels'] = ann_info['gt_labels'].copy()

        return results


@PIPELINES.register_module()
class TianchiFormatBundle(object):

    def __call__(self, results):
        results['img'] = []

        imgs = results['imgs']
        for img in imgs:
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['img'].append(img)
        results['img'] = DC(to_tensor(np.stack(results['img'])), stack=True)

        for key in ['valid', 'gt_points', 'gt_labels']:
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
        imgs = results['imgs']

        resized_imgs = []
        w_scales = []
        h_scales = []
        for _img in imgs:
            if self.keep_ratio:
                img, scale_factor = mmcv.imrescale(
                    _img, self.img_scale, return_scale=True)
                new_h, new_w = img.shape[:2]
                h, w = _img.shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            resized_imgs.append(img)
            w_scales.append(w_scale)
            h_scales.append(h_scale)

        results['imgs'] = resized_imgs
        results['img_shapes'] = [img.shape for img in resized_imgs]
        # in case that there is no padding
        results['pad_shapes'] = [img.shape for img in resized_imgs]
        results['scale_factors'] = np.transpose(np.stack((w_scales, h_scales)))
        results['keep_ratio'] = self.keep_ratio

    def _resize_points(self, results):
        gt_idx = results['gt_idx']
        scale_factor = results['scale_factors'][gt_idx]
        results['gt_points'] *= scale_factor

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

    def _pad_imgs(self, results):
        if self.size is not None:
            padded_imgs = [
                mmcv.impad(img, shape=self.size, pad_val=self.pad_val)
                for img in results['imgs']
            ]

        results['imgs'] = padded_imgs
        results['pad_shapes'] = [img.shape for img in padded_imgs]

    def __call__(self, resutls):
        self._pad_imgs(resutls)

        return resutls
