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
ANNS = {
    'lumbar_train150': 'annotations/lumbar_train150_annotation.json',
    'lumbar_train51': 'annotations/lumbar_train51_annotation.json',
}


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
    meta_keys = {
        'study_uid': '0020|000d',
        'series_uid': '0020|000e',
        'instance_uid': '0008|0018',
        'series_description': '0008|103e',
        'image_position': '0020|0032',
        'image_orientation': '0020|0037',
        'slice_thickness': '0018|0050',
        'pixel_spacing': '0028|0030',
    }

    _meta = {}
    for key, meta_key in meta_keys.items():
        if meta_key not in meta:
            continue
        _meta[key] = meta[meta_key]
    return _meta


def get_all_dicoms(study, reader, study_id, dicom_id, img_prefix):
    """
    Args:
        study (Path): study dir path.
        reader (sitk.Reader): DICOM file reader.
        study_id (int): 1-based study_id.
        dicom_id (int): 1-based dicom_id.
        img_prefix (Path): images dir path.

    Returns:
        study (dict): study annotations.
            - id
            - study_idx
            - study_uid
        dicom_list (list[dict]): dicom info list of the study.
        dicom_id: dicom counter
    """
    # study_idx is the original dirname in dataset
    study_idx = osp.basename(study)

    dicoms = glob(osp.join(study, '*.dcm'))
    dicom_list = []
    for dicom in dicoms:
        # read DICOM file
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
        img_path = osp.join(img_prefix, filename)
        if osp.exists(img_path):
            img = mmcv.imread(img_path)
        else:
            # no JPG img, convert DICOM to JPG
            img = load_dicom_image(reader)
            print(f'Writing {img_path}')
            mmcv.imwrite(img, img_path)
        height, width = img.shape[:2]
        _dicom_dict['height'] = height
        _dicom_dict['width'] = width

        dicom_list.append(_dicom_dict)
        dicom_id += 1

    # study info
    study_uid = dicom_list[-1]['study_uid']
    study = dict(id=study_id, study_idx=study_idx, study_uid=study_uid)

    return study, dicom_list, dicom_id


def parse_args():
    parser = argparse.ArgumentParser(
        description='Parse DICOM Meta and Generate Annotation File')
    parser.add_argument(
        '--data-root',
        default='/data/tianchi-lumbar',
        help='path to Tianchi lumbar dataset')
    parser.add_argument(
        '--dicom-prefix',
        default='lumbar_train150',
        help='prefix of DICOM file (dirname)')
    parser.add_argument(
        '--img-prefix', default=None, help='path of converted JPG images')
    parser.add_argument('--ann-file', default=None, help='annotation file')
    parser.add_argument(
        '--out-annfile',
        default=None,
        help='path to converted coco-style annotation file')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    data_root = args.data_root
    dicom_prefix = osp.join(data_root, args.dicom_prefix)
    print('This script is processing DICOM file dataset.')
    print(f'DICOM files in: {dicom_prefix}')

    with_anns = False
    if args.ann_file is None:
        if args.dicom_prefix in ANNS:
            # use predefined ann_file
            ann_file = ANNS[args.dicom_prefix]
            ann_file = osp.join(data_root, ann_file)
            print(f'Using pre-defined annotation file: {ann_file}')
            with_anns = True
        else:
            print('Without specified annotation file, this script will '
                  'generate ann_file without ground truths.')
    else:
        ann_file = osp.join(data_root, args.ann_file)
        print(f'Using specified annotation file {ann_file}.')
        with_anns = True

    # output paths
    if args.out_annfile is None:
        # default: use ``annotations/{dicom_prefix``
        out_annfile = f'annotations/{args.dicom_prefix}.json'
    else:
        out_annfile = args.out_annfile
    out_annfile = osp.join(data_root, out_annfile)
    # remove existing out_annfile
    if osp.exists(out_annfile):
        print(f'NOTE: {out_annfile} exists, it will be removed')
        os.remove(out_annfile)

    if args.img_prefix is None:
        # default: use ``images/{dicom_prefix}``
        img_prefix = f'images/{args.dicom_prefix}'
    else:
        img_prefix = args.img_prefix
    img_prefix = osp.join(data_root, img_prefix)
    print(f'Set img prefix to {img_prefix}')
    # check the img_prefix dir; this script will convert DICOM to JPG
    # if not existing
    mmcv.mkdir_or_exist(img_prefix)

    # 1. first get all study dirnames
    studies = glob(osp.join(dicom_prefix, 'study*'))
    print(f'There are {len(studies)} study instances in {dicom_prefix}')

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
        # study_id starts from 1
        study_id = i + 1
        study, _dicom_list, dicom_id = get_all_dicoms(study, reader, study_id,
                                                      dicom_id, img_prefix)

        study_dict[study['study_uid']] = study
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
    if with_anns:
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
    study_list = [study for _, study in study_dict.items()]
    coco_style_ann = dict(
        studies=study_list, dicoms=dicom_list, categories=categories)
    if with_anns:
        coco_style_ann['annotations'] = ann_list
    print(f'Writing {out_annfile}')
    mmcv.dump(coco_style_ann, out_annfile)


if __name__ == '__main__':
    main()
