import mmcv
import torch
from mmdet.apis import init_detector, inference_detector
import numpy as np
import os

# 配置文件路径
config_file = r"work_dirs\faster-rcnn_r50_fpn_1x_voc\faster-rcnn_r50_fpn_1x_voc.py"
# 预训练模型路径
checkpoint_file = r"work_dirs\faster-rcnn_r50_fpn_1x_voc\epoch_10.pth"
# 输入图像路径
img_path = r"demo\in\2007_000822.jpg"

# 加载模型
model = init_detector(config_file, checkpoint_file, device="cuda:0")

# 输入
img = mmcv.imread(img_path)
img = torch.from_numpy(img)
img = img.unsqueeze(0).permute((0, 3, 1, 2)).cuda().float()

# 获取 RPN 结果
with torch.no_grad():
    features = model.extract_feat(img)
    rpn_outs = model.rpn_head(features)
breakpoint()
img_metas = [{"filename": img_path, "ori_shape": (img.shape[2], img.shape[3], 3)}]
proposal_list = model.roi_head.predict_bbox(features,img_metas, *rpn_outs, model.test_cfg)
"""Perform forward propagation of the bbox head and predict detection
results on the features of the upstream network.

Args:
    x (tuple[Tensor]): Feature maps of all scale level.
    batch_img_metas (list[dict]): List of image information.
    rpn_results_list (list[:obj:`InstanceData`]): List of region
        proposals.
    rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
    rescale (bool): If True, return boxes in original image space.
        Defaults to False.

Returns:
    list[:obj:`InstanceData`]: Detection results of each image
    after the post process.
    Each item usually contains following keys.

        - scores (Tensor): Classification scores, has a shape
            (num_instance, )
        - labels (Tensor): Labels of bboxes, has a shape
            (num_instances, ).
        - bboxes (Tensor): Has a shape (num_instances, 4),
            the last dimension 4 arrange as (x1, y1, x2, y2).
"""

# 保存 RPN 结果
for i, proposals in enumerate(proposal_list):
    img_meta = img_metas[i]
    img_name = os.path.basename(img_meta["filename"])
    proposals_np = proposals.cpu().numpy()
    np.savetxt(f"{img_name}_rpn_proposals.txt", proposals_np, fmt="%f")
