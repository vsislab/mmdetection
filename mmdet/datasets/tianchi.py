import math
import os.path as osp

import mmcv
import numpy as np

from mmdet.core import Study
from .builder import DATASETS
from .pipelines import Compose

study_uid2study_idx = mmcv.load('mmdet/core/structure/study_uids.json')


@DATASETS.register_module()
class TianchiDataset(object):

    def __init__(self,
                 img_prefix,
                 pipeline,
                 ann_file=None,
                 data_root=None,
                 num_frames=4,
                 workers_per_gpu=8,
                 test_mode=False):
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix

        # join paths if data_root is specified
        if self.data_root is not None:
            if not (self.ann_file is None or osp.isabs(self.ann_file)):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)

        self.num_frames = num_frames
        self.workers_per_gpu = workers_per_gpu
        self.test_mode = test_mode

        if not self.test_mode:
            self.data_infos = self.load_annotations(self.ann_file)
        else:
            raise NotImplementedError

        # processing pipeline
        self.pipeline = Compose(pipeline)

    def load_annotations(self, ann_file):
        return mmcv.load(ann_file)

    def __len__(self):
        return len(self.data_infos)

    def _rand_another(self, idx):
        return np.random.randint(len(self))

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_study(idx)
        while True:
            data = self.prepare_train_study(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def get_ann_info(self, study, data_info):
        """Get annotation for training, by calling study.load_annotation().
        See ``mmdet/core/structure/study.py``
        """
        annotation = data_info['data'][0]['annotation'][0]['data']
        ann = study.load_annotation(annotation)

        return ann

    def prepare_train_study(self, idx):
        data_info = self.data_infos[idx]

        study_uid = data_info['studyUid']
        study_idx = study_uid2study_idx[study_uid]
        study = Study(
            osp.join(self.img_prefix, study_idx),
            num_processes=self.workers_per_gpu)

        # use the ``seriesUid``-specified Series (should be T2 & SAG)
        series_uid = data_info['data'][0]['seriesUid']
        series = study.series_dict[series_uid]
        if len(series) < self.num_frames:
            raise ValueError(f'There are only {len(series)} frames in series, '
                             f'but {self.num_frames} needed')

        # set the mid frame of series based on ``instanceUid``
        mid_frame_idx = series.set_mid_frame(
            data_info['data'][0]['instanceUid'])
        # get num_frames DICOM images around the mid_frame_idx
        n_before = math.ceil(self.num_frames / 2.)
        # the last not included
        n_after = int(self.num_frames - n_before)
        # return list of dicoms
        dicom = series[mid_frame_idx - n_before:mid_frame_idx + n_after]
        gt_idx = n_before

        ann_info = self.get_ann_info(study, data_info)

        results = dict(
            study=study,
            series=series,
            dicom=dicom,
            gt_idx=gt_idx,
            ann_info=ann_info)

        return self.pipeline(results)

    def prepare_test_study(self, idx):
        raise NotImplementedError
