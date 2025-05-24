_base_ = [
    '../_base_/models/sparse-rcnn_r50_fpn.py',
    '../_base_/datasets/voc0712.py',
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]

train_cfg = dict(max_epochs=100, val_interval=7)
device = 'cuda'

# model = dict(
#     roi_head=dict(
#         bbox_head=dict(num_classes=20),
#         mask_head=dict(num_classes=20)
#     )
# )

# # 数据集相关信息 (跟 VOC 相关)
# data_root = 'VOCdevkit/'
# metainfo = {
#     'classes': ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
#                 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
#                 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
#                 'train', 'tvmonitor'),
#     'palette': [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
#                 (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
#                 (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
#                 (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
#                 (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157)]
# }

# train_dataloader = dict(
#     dataset=dict(
#         data_root=data_root,
#         metainfo=metainfo,
#         ann_file='VOC2007/ImageSets/Main/trainval.txt',
#         data_prefix=dict(img='VOC2007/')
#     )
# )

# val_dataloader = dict(
#     dataset=dict(
#         data_root=data_root,
#         metainfo=metainfo,
#         ann_file='VOC2007/ImageSets/Main/val.txt',
#         data_prefix=dict(img='VOC2007/')
#     )
# )

# test_dataloader = val_dataloader

# # 加载预训练模型
# load_from = 'checkpoints/sparse-rcnn_r50_fpn_1x_voc0712.pth'
