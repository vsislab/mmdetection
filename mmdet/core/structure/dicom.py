import os.path as osp
import SimpleITK as sitk

import mmcv
import numpy as np


class DICOM(object):
    """DICOM class for Tianchi Lumbar Dataset.

    Available Tags:
        * https://dicom.innolitics.com/ciods/mr-image
        * Number of readable DICOMs in dataset: 8271
        '0008|0008': Image Type;                8271
        '0008|0016': SOP Class UID;             8271
        '0008|0018': SOP Instance UID;          8271
        '0008|0020': Study Date;                8271
        '0008|0021': Series Date;               8247
        '0008|0022': Acquisition Date;          7156
        '0008|0023': Content Date;              8052
        '0008|0030': Study Time;                8271
        '0008|0031': Series Time;               8247
        '0008|0032': Acquisition Time;          8039
        '0008|0033': Content Timee;             8052
        '0008|0060': Modality;                  8271
        '0008|0070': Manufacturer;              8271
        '0008|0080': Institution Name;          8271
        '0008|1010': Station Name;              8097
        '0008|103e': Series Description;        8271
        '0008|1090': Manufacturer's Model Name; 7947
        '0010|0010': Patient's Name;            8271
        '0010|0020': Patient ID;                8271
        '0010|0030': Patient's Birth Date;      8271
        '0010|0040': Patient's Sex;             8271
        '0010|1010': Patient's Age;             8271
        '0010|1030': Patient's Weight;          7742
        '0018|0015': Body Part Examined;        7326
        '0018|0050': Slice Thickness;           8247
        '0018|0087': Magnetic Field Strength;   7491
        '0018|0088': Spacing Between Slices;    8247
        '0018|0095': Pixel Bandwidth;           6509
        '0018|1000': Device Serial Number;      7364
        '0018|1020': Software Versions;         7749
        '0018|1310': Acquisition Matrix;        8247
        '0018|1312':
            In-plane Phase Encoding Direction;  6782
        '0018|1314': Flip Angle;                7538
        '0018|1316': SAR;                       7040
        '0020|000d': Study Instance UID;        8271
        '0020|000e': Series Instance UID;       8271
        '0020|0010': Study ID;                  8271
        '0020|0011': Series Number;             8271
        '0020|0012': Acquisition Number;        7867
        '0020|0013': Instance Number;           8271
        '0020|0032': Image Position (Patient);  8247
        '0020|0037':
            Image Orientation (Patient);        8247
        '0020|0052': Frame of Reference UID;    8247
        '0020|0060': Laterality;                6490
        '0020|1041': Slice Location;            2415
        '0020|1042': (Unknown);                  709
        '0028|0002': Samples per Pixel;         8271
        '0028|0004':
            Photometric Interpretation;         8271
        '0028|0010': Rows;                      8271
        '0028|0011': Columns;                   8271
        '0028|0030': Pixel Spacing;             8247
        '0028|0100': Bits Allocated;            8271
        '0028|0101': Bits Stored;               8271
        '0028|0102': High Bit;                  8271
        '0028|0103': Pixel Representation;      8271
        '0028|1050': Window Center;             7553
        '0028|1051': Window Width;              7553
        '0028|1052': Rescale Intercept;          850
        '0028|1053': Rescale Slope;              850
        '0028|1054': Rescale Type;               796
        '0040|0253':
            Performed Procedure Step ID;         667

    Args:
        path (str): DICOM file path
    """

    def __init__(self, path):
        reader = sitk.ImageFileReader()
        reader.LoadPrivateTagsOn()
        reader.SetImageIO('GDCMImageIO')
        reader.SetFileName(path)

        try:
            self.init_dicom(reader)
            self.readable = True
            # some dicom files have no `image_position`
            # and `image_orientation` attrs, set it to
            # unreadable
            if '0020|0032' not in self.meta or '0020|0037' not in self.meta:
                self.readable = False
        except RuntimeError:
            # some dicom files in dataset are unreadable
            self.readable = False

        # load basic info if readable
        if self.readable:
            self._fields = {}

            filename = osp.basename(path)
            # study_idx is the `study***` in dataset, not study uid
            study_idx = path.split('/')[-2]
            self._fields['filename'] = filename
            self._fields['study_idx'] = study_idx
            self._fields['study_uid'] = self.meta['0020|000d']
            self._fields['series_uid'] = self.meta['0020|000e']
            self._fields['instance_uid'] = self.meta['0008|0018']

    @property
    def filename(self):
        return self._fields['filename']

    @property
    def study_idx(self):
        return self._fields['study_idx']

    @property
    def study_uid(self):
        return self._fields['study_uid']

    @property
    def series_uid(self):
        return self._fields['series_uid']

    @property
    def instance_uid(self):
        return self._fields['instance_uid']

    @property
    def position(self):
        return self._fields['position']

    @property
    def orientation(self):
        return self._fields['orientation']

    @property
    def row_orientation(self):
        return self._fields['orientation'][:3]

    @property
    def col_orientation(self):
        return self._fields['orientation'][3:]

    @property
    def unit_normal_vector(self):
        a0, b0, c0 = self.row_orientation
        a1, b1, c1 = self.col_orientation

        y = (a1 * c0 - a0 * c1) / (b0 * c1 - b1 * c0 + 1e-12)
        z = (a0 * b1 - a1 * b0) / (b0 * c1 - b1 * c0 + 1e-12)
        normal_vector = np.array([1.0, y, z], dtype=np.float32)

        return normal_vector / np.linalg.norm(normal_vector)

    @property
    def plane(self):
        normal_vector = self.unit_normal_vector

        unit_x = np.array([1., 0., 0.], dtype=np.float32)
        unit_y = np.array([0., 1., 0.], dtype=np.float32)
        unit_z = np.array([0., 0., 1.], dtype=np.float32)

        if np.abs(np.matmul(normal_vector, unit_x)) > 0.866:
            return 'sag'
        elif np.abs(np.matmul(normal_vector, unit_y)) > 0.866:
            return 'cor'
        elif np.abs(np.matmul(normal_vector, unit_z)) > 0.866:
            return 'tra'
        else:
            return 'unknown'

    @property
    def series_description(self):
        return self._fields['series_description']

    @property
    def t_type(self):
        if 'T2' in self._fields['series_description'].upper():
            return 'T2'
        elif 'T1' in self._fields['series_description'].upper():
            return 'T1'
        else:
            return 'UNKNOWN'

    @property
    def image(self):
        return self._image

    def init_dicom(self, reader):
        reader.ReadImageInformation()
        self.meta = {
            key: reader.GetMetaData(key)
            for key in reader.GetMetaDataKeys()
        }

        # load image
        image = reader.Execute()
        image = sitk.GetArrayFromImage(image)[0]
        image = image.astype(np.float32)
        # output_pixel = (input_pixel - input_min) * \
        #   ((output_max - output_min) / (input_max - input_min)) + \
        #   output_min
        image = (image - image.min()) * (255.0 / (image.max() - image.min()))
        if self.meta['0028|0004'].strip() == 'MONOCHROME1':
            # If MONOCHROME2 (minimum should be displayed as black) we
            # don't need to do anything, if image has MONOCRHOME1
            # (minimum should be displayed as white) we flip # the intensities.
            # This is a constraint imposed by ITK which always assumes
            # MONOCHROME2.
            image = 255.0 - image
        self._image = np.ascontiguousarray(image)

    def parse_meta(self):
        # series description seems to be the only useful metadata
        # to know that the dicom is a T1 image or T2 image for now
        series_description = self.meta['0008|103e']
        series_description = series_description.encode(
            'utf-8', 'ignore').decode('utf-8')
        self._fields['series_description'] = series_description

        # image position (0020,0032) specifies the x, y, and z coordinates
        # of the upper left hand corner of the image
        image_position = self.meta['0020|0032'].strip().split('\\')
        assert len(image_position) == 3, 'Bad DICOM: len(Image Position) != 3'
        image_position = np.array(image_position, dtype=np.float32)
        self._fields['position'] = image_position

        # image orientation (0020,0037) specifies the direction cosines
        # of the first row and the first column with respect to the patient
        image_orientation = self.meta['0020|0037'].strip().split('\\')
        assert len(
            image_orientation) == 6, 'Bad DICOM: len(Image Orientation) != 6'
        image_orientation = np.array(image_orientation, dtype=np.float32)
        self._fields['orientation'] = image_orientation

    def visualization(self,
                      out=None,
                      out_dir='work_dirs/visualizations',
                      image=None):
        """
        Args:
            out: output filename. Defaults to None.
            out_dir: output path (dir) to save visualization.
            image: image to be visualized.
                Defaults to None (original image, self.image).
                Will be specified if visualized with ground truths.
        """
        image = self.image if image is None else image

        if out is not None:
            mmcv.imwrite(image, out)
        else:
            # output filename will be study_idx + filename
            filename = self.filename.replace('.dcm', '.jpg')
            filename = f'{self.study_idx}_{filename}'
            mmcv.imwrite(image, osp.join(out_dir, filename))

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(instance_uid={self.instance_uid},'
        repr_str += f' plane={self.plane})'

        return repr_str
