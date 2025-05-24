from mmdet.registry import DATASETS
from mmdet.datasets.xml_style import XMLDataset
import os.path as osp
import mmengine.fileio as fileio
import mmcv
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image


@DATASETS.register_module()
class VOCInstanceSegDataset(XMLDataset):
    """VOC dataset for instance segmentation.

    Annotations are in PASCAL VOC format, and segmentation masks are in
    `SegmentationObject/` directory.
    """

    METAINFO = {
        'classes':
        ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
         'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
         'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
    }

    def load_data_list(self):
        data_list = []
        for img_id in self.data_infos:
            img_path = osp.join(self.data_prefix['img_path'], img_id['filename'])
            ann_path = osp.join(self.data_prefix['ann_path'], img_id['ann'])

            # segmentation mask路径
            seg_map_path = osp.join(self.data_prefix['seg_map_path'], img_id['filename'].replace('.jpg', '.png'))

            # 读取 XML
            raw_ann_info = ET.parse(ann_path).getroot()
            instances = []
            for obj in raw_ann_info.findall('object'):
                name = obj.find('name').text
                if name not in self.METAINFO['classes']:
                    continue
                label = self.METAINFO['classes'].index(name)

                bbox = obj.find('bndbox')
                bbox = [
                    float(bbox.find('xmin').text),
                    float(bbox.find('ymin').text),
                    float(bbox.find('xmax').text),
                    float(bbox.find('ymax').text)
                ]

                instance_id = int(obj.find('segm').text) if obj.find('segm') is not None else None
                instances.append(dict(bbox=bbox, label=label, instance_id=instance_id))

            # 读取 mask PNG 图像
            if osp.exists(seg_map_path):
                mask_img = np.array(Image.open(seg_map_path))
                for inst in instances:
                    if inst['instance_id'] is None:
                        continue
                    mask = (mask_img == inst['instance_id']).astype(np.uint8)
                    inst['mask'] = mask
            else:
                # 如果 mask 路径缺失，跳过该样本或报错
                continue

            data_list.append(
                dict(
                    img_path=img_path,
                    instances=[{
                        'bbox': inst['bbox'],
                        'bbox_label': inst['label'],
                        'mask': inst['mask']
                    } for inst in instances if 'mask' in inst]
                )
            )
        return data_list
