import argparse
import os
import os.path as osp
import SimpleITK as sitk
from collections import defaultdict
from glob import glob

import mmcv
import numpy as np

DISC_PARTS = ('T11-T12', 'T12-L1', 'L1-L2', 'L2-L3', 'L3-L4', 'L4-L5', 'L5-S1')
VERTEBRA_PARTS = ('L1', 'L2', 'L3', 'L4', 'L5')


def get_disc_id(tag):
    tags = ['v1', 'v2', 'v3', 'v4', 'v5']
    assert tag in tags

    return tags.index(tag) + 1


def get_vertebra_id(tag):
    tags = ['v1', 'v2']
    assert tag in tags

    return tags.index(tag) + 1 + 5


def load_dicom_image(reader):
    image = reader.Execute()
    image = sitk.GetArrayFromImage(image)[0]
    image = image.astype(np.float32)
    # output_pixel = (input_pixel - input_min) * \
    #   ((output_max - output_min) / (input_max - input_min)) + \
    #   output_min
    image = (image - image.min()) * (255.0 / (image.max() - image.min()))
    if reader.GetMetaData('0028|0004').strip() == 'MONOCHROME1':
        # If MONOCHROME2 (minimum should be displayed as black) we
        # don't need to do anything, if image has MONOCRHOME1
        # (minimum should be displayed as white) we flip # the intensities.
        # This is a constraint imposed by ITK which always assumes
        # MONOCHROME2.
        image = 255.0 - image

    return np.ascontiguousarray(image)


def parse_meta(meta):
    study_uid = meta['0020|000d']
    series_uid = meta['0020|000e']
    instance_uid = meta['0008|0018']
    series_description = meta['0008|103e'].strip()

    meta_info = dict(
        study_uid=study_uid,
        series_uid=series_uid,
        instance_uid=instance_uid,
        serise_description=series_description)

    keynames = [
        'image_position', 'image_orientation', 'slice_thickness',
        'pixel_spacing'
    ]
    meta_keys = ['0020|0032', '0020|0037', '0018|0050', '0028|0030']
    for keyname, key in zip(keynames, meta_keys):
        if key not in meta:
            continue
        value = meta[key].strip().split('\\')
        value = list(map(float, value))
        meta_info[keyname] = value

    return meta_info


def parse_args():
    parser = argparse.ArgumentParser(
        description='Parse DICOM Meta and Generate Annotation File')
    parser.add_argument(
        '--data-root',
        default='/data/tianchi-lumbar',
        help='path to Tianchi lumbar dataset')
    parser.add_argument(
        '--img-prefix',
        default='lumbar_train150',
        help='prefix of DICOM file (dirname)')
    parser.add_argument(
        '--ann-file',
        default='annotations/lumbar_train150_annotation.json',
        help='annotation file')
    parser.add_argument(
        '--out-annfile',
        default='annotations/lumbar_train150.json',
        help='path to converted coco-style annotation file')
    parser.add_argument(
        '--out-dir',
        default='train150',
        help='path of output converted images')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    data_root = args.data_root
    img_prefix = osp.join(data_root, args.img_prefix)
    ann_file = osp.join(data_root, args.ann_file)
    out_annfile = osp.join(data_root, args.out_annfile)
    out_dir = osp.join(data_root, args.out_dir)

    if osp.exists(out_annfile):
        print(f'NOTE: {out_annfile} exists, it will be removed')
        os.remove(out_annfile)

    # check the out_dir; without JPG images, this script will
    # convert DICOM to JPG
    mmcv.mkdir_or_exist(out_dir)

    # 1. first get all study dirnames
    studies = glob(osp.join(img_prefix, 'study*'))
    print(f'There are {len(studies)} study instances in {img_prefix}')

    # 2. set DICOM file reader
    reader = sitk.ImageFileReader()
    reader.LoadPrivateTagsOn()
    reader.SetImageIO('GDCMImageIO')

    # 3. get all DICOMs and studies
    dicom_id = 1  # start from 1
    dicom_list = []
    # keep a dict (study_uid as key) for adding annotationed info
    # it will be converted to list when writing new annotation files
    study_dict = defaultdict(dict)
    for i, study in enumerate(studies):
        # study_idx is the original dirname in dataset
        study_idx = osp.basename(study)
        # study_id starts from 1
        study_id = i + 1

        dicoms = glob(osp.join(study, '*.dcm'))
        _dicom_list = []
        for dicom in dicoms:
            reader.SetFileName(dicom)
            try:
                reader.ReadImageInformation()
            except RuntimeError:
                continue

            _dicom_dict = dict(id=dicom_id, study_id=study_id)
            filename = osp.basename(dicom)
            filename = filename.replace('.dcm', '.jpg')
            filename = f'{study_idx}_{filename}'
            _dicom_dict['filename'] = filename

            meta = {
                key: reader.GetMetaData(key)
                for key in reader.GetMetaDataKeys()
            }
            _dicom_dict.update(parse_meta(meta))

            # image height and width
            _filename = osp.join(out_dir, filename)
            if osp.exists(_filename):
                image = mmcv.imread(_filename)
            else:
                # no JPG image, convert DICOM to JPG
                image = load_dicom_image(reader)
                print(f'Writing {_filename}')
                mmcv.imwrite(image, _filename)
            height, width = image.shape[:2]
            _dicom_dict['height'] = height
            _dicom_dict['width'] = width

            _dicom_list.append(_dicom_dict)
            dicom_id += 1

        study_uid = _dicom_list[-1]['study_uid']
        study = dict(id=study_id, study_idx=study_idx, study_uid=study_uid)
        study_dict[study_uid] = study
        dicom_list.extend(_dicom_list)

    # 4. get all categories
    categories = [
        dict(id=1, name='disc', tag='v1'),
        dict(id=2, name='disc', tag='v2'),
        dict(id=3, name='disc', tag='v3'),
        dict(id=4, name='disc', tag='v4'),
        dict(id=5, name='disc', tag='v5'),
        dict(id=6, name='vertebra', tag='v1'),
        dict(id=7, name='vertebra', tag='v2')
    ]

    # 5. get annotations if exists
    if ann_file != 'none':
        print(f'Ground-truth annotation file {ann_file} exists.')
        print(f'Converting original annotations to {out_annfile}')

        ori_anns = mmcv.load(ann_file)
        ann_id = 1  # start from 1
        ann_list = []

        for i, _ann in enumerate(ori_anns):
            # study info in original annotations
            study_uid = _ann['studyUid']
            series_uid = _ann['data'][0]['seriesUid']
            instance_uid = _ann['data'][0]['instanceUid']
            study_dict[study_uid].update(
                dict(series_uid=series_uid, instance_uid=instance_uid))
            study_id = study_dict[study_uid]['id']

            # annotations of each study
            _ann_list = []
            points = _ann['data'][0]['annotation'][0]['data']['point']
            for _point in points:
                part = _point['tag']['identification']
                if part in DISC_PARTS:
                    assert 'disc' in _point['tag']
                    tag = _point['tag']['disc']
                    if tag == '':
                        tag = 'v1'  # labeled as normal if missed
                    if ',' in tag:
                        # some disc have multi-labels
                        for _tag in tag.split(','):
                            cat_id = get_disc_id(_tag)
                            _ann_list.append(
                                dict(
                                    id=ann_id,
                                    study_id=study_id,
                                    category_id=cat_id,
                                    point=_point['coord'],
                                    identification=part))
                            ann_id += 1
                    else:
                        cat_id = get_disc_id(tag)
                        _ann_list.append(
                            dict(
                                id=ann_id,
                                study_id=study_id,
                                category_id=cat_id,
                                point=_point['coord'],
                                identification=part))
                        ann_id += 1
                elif part in VERTEBRA_PARTS:
                    assert 'vertebra' in _point['tag']
                    tag = _point['tag']['vertebra']
                    if tag == '':
                        tag = 'v1'  # labeled as normal if missed
                    cat_id = get_vertebra_id(tag)
                    _ann_list.append(
                        dict(
                            id=ann_id,
                            study_id=study_id,
                            category_id=cat_id,
                            point=_point['coord'],
                            identification=part))
                    ann_id += 1

            ann_list.extend(_ann_list)

    # 6. write to out_annfile
    study_list = [study for uid, study in study_dict.items()]
    coco_style_ann = dict(
        studies=study_list, dicoms=dicom_list, categories=categories)
    if ann_list != 'none':
        coco_style_ann['annotations'] = ann_list
    print(f'Writing annotations to {out_annfile}')
    mmcv.dump(coco_style_ann, out_annfile)


if __name__ == '__main__':
    main()
