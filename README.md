# voc-mmdetection
Mid-term project of DATA620004: Training Mask R-CNN and Sparse R-CNN on the VOC2012 dataset with mmdetection.

This project is based on [MMDetection](https://github.com/open-mmlab/mmdetection), 
which is released under the [Apache License 2.0](LICENSE).

# 🚀 Object Detection on VOC2012 with MMDetection

This project perform object detection on the Pascal VOC 2012 dataset using models such as Mask R-CNN and Sparse R-CNN.

---

## 📦 Environment Setup

We will use `mmdetection` for this task.

### 1. Install PyTorch (CUDA 11.7)

Due to compatibility with `mmdetection`, install `torch==2.0.1` with CUDA 11.7:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```


##  Install MMDetection via OpenMIM
```bash
pip3 install -U openmim
mim install mmengine
mim install "mmcv==2.1.0"
mim install mmdet
```

## 📥 Dataset Download
Download the VOC2012 dataset using the provided script:

```bash
python tools/misc/download_dataset.py --dataset-name voc2012
```





## 🗂️ Dataset Organization
After downloading, extract the data into the data folder. The structure should look like:

```css
data
└─coco
    └─VOCtrainval_11-May-2012
        └─VOCdevkit
            └─VOC2012
                ├─Annotations
                ├─ImageSets
                │  ├─Action
                │  ├─Layout
                │  ├─Main
                │  └─Segmentation
                ├─JPEGImages
                ├─SegmentationClass
                └─SegmentationObject
```
Total images: 17,125

## 🧪 Dataset Splits
We use the following for training and validation:

Training set (5,717 images):

```swift
data/coco/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Main/train.txt
```
Validation set (5,717 images):

```swift
data/coco/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Main/val.txt
```

## 🏋️‍♂️ Model Training
Use MMDetection’s training scripts.

Mask R-CNN
Train from scratch:

```bash
python tools/train.py configs/mask-rcnn_r50_fpn_1x_voc.py
```
Resume from checkpoint:

```bash
python tools/train.py configs/mask-rcnn_r50_fpn_1x_voc.py --resume [checkpoint.pth]
```

Sparse R-CNN
Train from scratch:

```bash
python tools/train.py configs/sparse_rcnn/sparse-rcnn_r50_fpn_1x_voc.py
```
Resume from checkpoint:

```bash
python tools/train.py configs/sparse_rcnn/sparse-rcnn_r50_fpn_1x_voc.py --resume [checkpoint.pth]
```
Training logs and weights will be saved to work_dirs/.

## 🔍 Model Testing
Use the following commands to test the trained models:

Mask R-CNN
```bash
python tools/test.py configs/mask-rcnn_r50_fpn_1x_voc.py --resume [checkpoint.pth]
```
Sparse R-CNN
```bash
python tools/test.py configs/sparse_rcnn/sparse-rcnn_r50_fpn_1x_voc.py --resume [checkpoint.pth]
```
## 🧠 Model Inference
Run inference on a custom image:

```bash
python inference.py
```



## 📦 Pretrained Weights

You can download the pretrained Mask R-CNN weights from Baidu NetDisk:

- [Mask R-CNN](https://pan.baidu.com/s/1canbZ35cyiCUD3mD-a1KtA?pwd=fsqn)
- **Extraction Code**: `fsqn`


- [Sparse R-CNN](https://drive.google.com/file/d/1kpI2g3wCGNX0LC8PU89CO_K-TxAJ3gnO/view?usp=sharing)


