import os
import cv2
import json

json_dir='D:/homework/MLZL/voc-mmdetection-main/data/coco/annotations'
train_path=os.path.join(json_dir,'train.json')
val_path=os.path.join(json_dir,'val.json')

img_path='D:/homework/MLZL/voc-mmdetection-main/data/VOCdevkit/VOC2012/JPEGImages'
train_dir = 'D:/homework/MLZL/voc-mmdetection-main/data/coco/train2017'
val_dir = 'D:/homework/MLZL/voc-mmdetection-main/data/coco/val2017'

# 确保输出目录存在
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

trainimg = json.load(open(train_path))['images']
print(f"Found {len(trainimg)} training images.")
valimg = json.load(open(val_path))['images']
print(f"Found {len(valimg)} validation images.")

for img in trainimg:
    img_name = img['file_name']
    full_img_path = os.path.join(img_path, os.path.basename(img_name))  # 只取文件名
    load_img = cv2.imread(full_img_path)
    if load_img is None:
        print(f"[WARN] Failed to load image: {full_img_path}")
        continue
    cv2.imwrite(os.path.join(train_dir, os.path.basename(img_name)), load_img)

for img in valimg:
    img_name = img['file_name']
    full_img_path = os.path.join(img_path, os.path.basename(img_name))  # 只取文件名
    load_img = cv2.imread(full_img_path)
    if load_img is None:
        print(f"[WARN] Failed to load image: {full_img_path}")
        continue
    cv2.imwrite(os.path.join(val_dir, os.path.basename(img_name)), load_img)

#python split.py