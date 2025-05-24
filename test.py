import os

data_root = 'data/coco/VOCtrainval_11-May-2012/VOCdevkit/VOC2007/'
ann_path = os.path.join(data_root, 'Annotations')
file_id = '000012'
xml_path = os.path.join(ann_path, file_id + '.xml')
xml_path = xml_path.replace(os.path.sep, '//') 
print("XML path:", xml_path)
print("Exists:", os.path.exists(xml_path))
