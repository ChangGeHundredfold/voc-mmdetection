import os
import mmengine
import numpy as np
from PIL import Image
from mmdet.registry import DATASETS
from mmdet.datasets import XMLDataset
import os.path as osp
import xml.etree.ElementTree as ET
from pycocotools import mask as maskUtils
from concurrent.futures import ThreadPoolExecutor, as_completed
print("custom_datasets.py has been successfully imported.")
def calculate_bbox(mask):
    """Calculate the bounding box from a binary mask."""
    coords = np.argwhere(mask)
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0)
    return [x0, y0, x1, y1]

@DATASETS.register_module()
class VOCInstanceDataset(XMLDataset):
    """VOC dataset with instance segmentation support from SegmentationObject."""

    METAINFO = {
        'classes': (
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'),
        'palette': [
            (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32),
            (0, 0, 142), (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 20, 60),
            (255, 0, 0), (0, 0, 0), (70, 130, 180), (0, 0, 70), (0, 0, 142),
            (0, 0, 192), (152, 251, 152), (220, 220, 0), (107, 142, 35), (0, 255, 255)
        ]
    }
    def __init__(self, seg_prefix=None, num_workers=4, **kwargs):
        self.seg_prefix = seg_prefix
        self.num_workers = num_workers
        super().__init__(**kwargs)
    def load_data_list(self) -> list:
        """Load the annotation and segmentation info, and add masks to instances."""
        print("data_root:", self.data_root)
        print("Loading data list...")
        
        data_list = super().load_data_list()
        print("Data list loaded successfully.")
        
        
        seg_prefix = self.data_prefix.get('seg_map_path', None)
        if seg_prefix is None:
            raise ValueError("Missing 'seg_map_path' in data_prefix")

        for item in data_list:
            item['img_id'] = osp.splitext(osp.basename(item['img_path']))[0]
            seg_file = osp.join(seg_prefix, item['img_id'] + '.png')

            if not osp.exists(seg_file):
                raise FileNotFoundError(f'Segmentation mask not found: {seg_file}')

            item['seg_map'] = seg_file

        print("Finished processing data list.")
        return data_list    


    def parse_data_info(self, raw_data_info):
        data_info = super().parse_data_info(raw_data_info)
        img_id = raw_data_info['img_id']
        ann_dir = osp.join(self.data_root, self.ann_subdir)
        xml_path = osp.join(ann_dir, f'{img_id}.xml')

        
        # Ensure 'ann' key exists in data_info
        if 'ann' not in data_info:
            data_info['ann'] = {}

        # Set the file path for annotations
        data_info['ann']['file_path'] = xml_path
        seg_prefix = self.data_prefix.get('seg_map_path', None)
        if seg_prefix is not None:
            seg_path = osp.join(seg_prefix, f'{img_id}.png')
        else:
            seg_path = None
        # Get per-object masks
        instances = []
        if seg_path is not None and osp.exists(seg_path):
            mask = np.array(Image.open(seg_path))
            tree = ET.parse(xml_path)
            root = tree.getroot()
            objs = root.findall('object')
            for idx, obj in enumerate(objs):
                name = obj.find('name').text
                cls_id = self.metainfo['classes'].index(name)
                # Each object has instance id = idx + 1
                inst_mask = (mask == (idx + 1)).astype(np.uint8)
                if inst_mask.sum() == 0:
                    continue
                bbox = calculate_bbox(inst_mask)  # Calculate bounding box
                rle_mask = self.mask_to_rle(inst_mask)
                instances.append({
                    'mask': rle_mask,  # Use RLE encoding for mask
                    'bbox': bbox,
                    'bbox_label': cls_id,  # Add bbox_label key
                    'ignore_flag': False  # Add ignore_flag key
                })

        if instances:
            data_info['instances'] = instances 
        return data_info
    def mask_to_rle(self, mask):
        """Convert a binary mask to RLE format using pycocotools."""
        rle = maskUtils.encode(np.asfortranarray(mask))
        return {'counts': rle['counts'].decode('utf-8'), 'size': list(mask.shape)}