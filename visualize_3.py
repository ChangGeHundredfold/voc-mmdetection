import os
import mmcv
import numpy as np
import cv2
from mmdet.apis import init_detector, inference_detector

config_file = 'work_dirs/mask-rcnn_r50_fpn_1x_voc/mask-rcnn_r50_fpn_1x_voc.py'
checkpoint_file = 'work_dirs/mask-rcnn_r50_fpn_1x_voc/epoch_10.pth'
img_dir = 'demo/show'
output_dir = 'outputs/vis_3'
os.makedirs(output_dir, exist_ok=True)

# 获取类别名
model = init_detector(config_file, checkpoint_file, device='cuda:0')
class_names = model.dataset_meta['classes']

img_list = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

for img_path in img_list:
    img = mmcv.imread(img_path)
    result = inference_detector(model, img_path)
    pred_instances = result.pred_instances
    bboxes = pred_instances.bboxes.cpu().numpy()
    labels = pred_instances.labels.cpu().numpy()
    scores = pred_instances.scores.cpu().numpy()
    masks = pred_instances.masks.cpu().numpy() if hasattr(pred_instances, 'masks') and pred_instances.masks is not None else None

    # 绘制检测框、掩码、类别和得分
    vis_img = img.copy()
    color_map = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
    for idx, bbox in enumerate(bboxes):
        if scores[idx] < 0.3:
            continue
        color = color_map[labels[idx] % len(color_map)]
        x1, y1, x2, y2 = bbox.astype(int)
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
        label_text = f'{class_names[labels[idx]]}: {scores[idx]:.2f}'
        cv2.putText(vis_img, label_text, (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        # 绘制掩码
        if masks is not None:
            mask = masks[idx].astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_img, contours, -1, color, 2)
            vis_img[mask > 0] = vis_img[mask > 0] * 0.5 + np.array(color) * 0.5

    out_path = os.path.join(output_dir, os.path.basename(img_path).replace('.', '_vis.'))
    mmcv.imwrite(vis_img, out_path)

print(f'可视化结果已保存到 {output_dir}')