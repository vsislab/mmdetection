import multiprocessing
import os.path as osp
from collections import defaultdict
from glob import glob

import numpy as np

from .dicom import DICOM
from .series import Series


class Study(object):
    """
    DISC_CLASSES:
        v1: Normal
        v2: Bulge
        v3: Protruded
        v4: Extruded
        v5: Schmor

    VERTEBRA_CLASSES:
        v1: Normal
        v2: Degeneration

    Args:
        path: path to study dir
        num_processes: number of processes when init DICOMs
    """
    PARTS = ('T12-L1', 'L1', 'L1-L2', 'L2', 'L2-L3', 'L3', 'L3-L4', 'L4',
             'L4-L5', 'L5', 'L5-S1')
    DISC_PARTS = ('T12-L1', 'L1-L2', 'L2-L3', 'L3-L4', 'L4-L5', 'L5-S1')
    DISC_CLASSES = ('v1', 'v2', 'v3', 'v4', 'v5')
    VERTEBRA_PARTS = ('L1', 'L2', 'L3', 'L4', 'L5')
    VERTEBRA_CLASSES = ('v1', 'v2')

    def __init__(self, path, num_processes=4):
        # get all DICOMs
        dicoms = glob(osp.join(path, '*.dcm'))
        if num_processes > 0:
            with multiprocessing.Pool(num_processes) as pool:
                dicoms = pool.map(DICOM, dicoms)
        else:
            dicoms = [DICOM(dicom) for dicom in dicoms]
        # Study will keep all dicoms in self.dicoms
        # but some dicoms will be abandoned in Series
        self.dicoms = [dicom for dicom in dicoms if dicom.readable]
        self.dicom_dict = {}
        # load more meta data and add to dicom_dict
        for dicom in self.dicoms:
            dicom.parse_meta()
            self.dicom_dict[dicom.instance_uid] = dicom

        # organize the DICOMs as a set of Series
        _series_dict = defaultdict(list)
        for dicom in self.dicoms:
            _series_dict[dicom.series_uid].append(dicom)
        # self.series: (list) of Series
        # self.series_dict: (dict) of Series, series_uid is key
        self.series = []
        self.series_dict = {}
        for series_uid, _series in _series_dict.items():
            series = Series(_series)
            self.series.append(series)
            self.series_dict[series_uid] = series

        self.T2_series_list = []
        self.T2_sag_series_list = []
        self.T2_tra_series_list = []
        for _series in self.series:
            if _series.t_type == 'T2':
                self.T2_series_list.append(_series)

                if _series.plane == 'sag':
                    self.T2_sag_series_list.append(_series)
                elif _series.plane == 'tra':
                    self.T2_tra_series_list.append(_series)

        # set up Study meta data
        self.study_idx = self.dicoms[0].study_idx
        self.study_uid = self.dicoms[0].study_uid

    @property
    def with_T2_series(self):
        return len(self.T2_series_list) > 0

    @property
    def with_T2_sag_series(self):
        return len(self.T2_sag_series_list) > 0

    @property
    def with_T2_tra_series(self):
        return len(self.T2_tra_series_list) > 0

    def _get_label(self, label, classes):
        labels = label.split(',')

        if len(labels) == 1:
            if label == '':
                label = 1
            else:
                label = labels[0][-1]
        elif len(labels) == 2:
            # DISC parts may have multi-labels, ``vX1, vX2``
            # choose the first one
            label = labels[0][-1]

        return int(label)

    def visualization(self, out_dir='work_dirs/visualizations'):
        for dicom in self.dicoms:
            dicom.visualization(out_dir=out_dir)

    def visualization_with_results(self, dicom_instance_uid, valid, points,
                                   labels):
        # TODO: better visualization with results implementation
        import mmcv
        import cv2

        dicom = self.dicom_dict[dicom_instance_uid]
        image = dicom.image
        image = mmcv.gray2bgr(image)

        for i in range(len(valid)):
            flag = valid[i]
            if flag:
                point = points[i, :]
                cv2.circle(image, tuple(point), 2, mmcv.color_val('red'))

        dicom.visualization(
            image=image,
            out=f'work_dirs/visualizations/with_results/{self.study_idx}.jpg')

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(study_idx={self.study_idx},'
        repr_str += f' study_uid={self.study_uid},'
        repr_str += f' num_series={len(self.series)},'
        repr_str += f' num_dicoms={len(self.dicoms)})'

        return repr_str
