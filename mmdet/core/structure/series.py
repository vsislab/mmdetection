import math
import os.path as osp
from collections import Counter

import numpy as np


class Series(object):
    plane_to_dim = {'sag': 0, 'cor': 1, 'tra': 2}

    def __init__(self, dicoms):
        # set up meta data
        self._study_idx = dicoms[0].study_idx
        self._study_uid = dicoms[0].study_uid
        self._series_uid = dicoms[0].series_uid
        self._series_description = dicoms[0].series_description
        self._t_type = dicoms[0].t_type

        # plane
        planes = Counter([dicom.plane for dicom in dicoms])
        self._plane = planes.most_common(1)[0][0]
        # only keep dicoms with same plane
        dicoms = [dicom for dicom in dicoms if dicom.plane == self._plane]
        # sort dicoms with positions
        positions = np.stack([dicom.position for dicom in dicoms])
        indices = np.argsort(positions[:, self.plane_to_dim[self._plane]])
        # keep the sorted positions
        self._positions = positions[indices]
        # keep sorted indices
        self._indices = indices

        # ugly sort
        self.dicoms = [dicoms[idx] for idx in self._indices]
        self._instance_uids = [dicom.instance_uid for dicom in self.dicoms]
        # set default mid frame idx
        # TODO: better mid frame setup
        self._mid_frame_idx = math.ceil(len(self.dicoms) / 2.)

    @property
    def study_idx(self):
        return self._study_idx

    @property
    def study_uid(self):
        return self._study_uid

    @property
    def series_uid(self):
        return self._series_uid

    @property
    def series_description(self):
        return self._series_description

    @property
    def t_type(self):
        return self._t_type

    @property
    def plane(self):
        return self._plane

    @property
    def positions(self):
        return self._positions

    @property
    def mid_frame_idx(self):
        return self._mid_frame_idx

    def __len__(self):
        return len(self.dicoms)

    def __iter__(self):
        return iter(self.dicoms)

    def __getitem__(self, idx):
        return self.dicoms[idx]

    def set_mid_frame(self, instance_uid):
        self._mid_frame_idx = self._instance_uids.index(instance_uid)

        return self._mid_frame_idx

    def visualization(self, out_dir='work_dirs/visualizations'):
        # out_dir = out_dir/study_idx/{t_type}_{plane}_{series_uid}
        _dirname = f'{self._t_type}_{self.plane}_{self._series_uid}'
        out_dir = osp.join(out_dir, self._study_idx, _dirname)
        # visualization with sorted indices
        for i, dicom in enumerate(self.dicoms):
            dicom.visualization(out=osp.join(out_dir, f'{i + 1}.jpg'))

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(series_uid={self.series_uid},'
        repr_str += f' series_description={self.series_description},'
        repr_str += f' plane={self.plane},'
        repr_str += f' t_type={self.t_type},'
        repr_str += f' num_dicoms={len(self.dicoms)})'

        return repr_str
