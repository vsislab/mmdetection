import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC

from ..builder import PIPELINES
from .formating import DefaultFormatBundle, to_tensor
from .loading import LoadAnnotations
from .transforms import Resize, RandomFlip


@PIPELINES.register_module()
class TianchiLoadAnnotation(LoadAnnotations):

    def __init__(self,
                 with_point=True,
                 with_bbox=True,
                 with_label=True,
                 with_part_inds=True):
        super().__init__(with_bbox, with_label)
        self.with_point = with_point
        self.with_part_inds = with_part_inds

    def _load_points(self, results):
        """Private function to load point annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box annotations.
        """

        ann_info = results['ann_info']
        results['gt_points'] = ann_info['points'].copy()
        results['point_fields'].append('gt_points')
        if self.with_part_inds:
            results['part_inds'] = results['ann_info']['part_insd'].copy()
        return results

    def __call__(self, results):
        results = super().__call__(results)
        if self.with_point:
            results = self._load_points

        return results


@PIPELINES.register_module()
class TianchiFormatBundle(DefaultFormatBundle):

    def __call__(self, results):
        results = super().__call__(results)
        for key in ['gt_points', 'part_inds']:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]))

        return results


@PIPELINES.register_module()
class TianchiResize(Resize):

    def _resize_points(self, results):
        """Resize points with ``results['scale_factor']``."""
        img_shape = results['img_shape']
        for key in results.get('point_fields', []):
            points = results[key] * results['scale_factor']
            points[:, 0] = np.clip(points[:, 0], 0, img_shape[1])
            points[:, 1] = np.clip(points[:, 1], 0, img_shape[0])
            results[key] = points

    def __call__(self, results):
        results = super().__call__(results)
        results = self._resize_points(results)

        return results


@PIPELINES.register_module()
class TianchiRandomFlip(RandomFlip):
    """Flip the image & point.
    """

    def point_flip(self, points, img_shape, direction):
        """Flip points.
        """

        assert points.shape[-1] == 2
        flipped = points.copy()
        if direction == 'horizontal':
            w = img_shape[1]
            flipped[..., 0] = w - points[..., 0]
        elif direction == 'vertical':
            h = img_shape[0]
            flipped[..., 1] = h - points[..., 1]
        else:
            raise ValueError(f"Invalid flipping direction '{direction}'")

        return flipped

    def __call__(self, results):
        results = super().__call__(results)
        if results['flip']:
            # flip points
            for key in results.get('point_fields', []):
                results[key] = self.point_flip(results[key],
                                               results['img_shape'],
                                               results['flip_direction'])
        return results


@PIPELINES.register_module()
class TianchiVisual(object):

    def __init__(self, with_gt=True, out_prefix='work_dirs/visualizations/'):
        self.with_gt = with_gt
        self.out_prefix = out_prefix

    def __call__(self, results):
        import cv2
        img = np.ascontiguousarray(results['img'])

        if self.with_gt:
            points = results['gt_points']
            bboxes = results['gt_bboxes']

            for i in range(points.shape[0]):
                point = tuple(map(int, points[i]))
                bbox = tuple(map(int, bboxes[i]))
                img = cv2.circle(img, point, 2, mmcv.color_val('red'))
                img = cv2.rectangle(img, bbox[:2], bbox[2:],
                                    mmcv.color_val('green'))

        filename = self.out_prefix + results['ori_filename']
        mmcv.imwrite(img, filename)

        return results
