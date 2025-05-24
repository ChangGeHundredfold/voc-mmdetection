import os
import mmcv
import numpy as np
import cv2
from mmdet.apis import init_detector, inference_detector

config_file = 'work_dirs/mask-rcnn_r50_fpn_1x_voc/mask-rcnn_r50_fpn_1x_voc.py'
checkpoint_file = 'work_dirs/mask-rcnn_r50_fpn_1x_voc/epoch_10.pth'
img_dir = 'demo/in'
output_dir = 'outputs/vis_2'
os.makedirs(output_dir, exist_ok=True)

model = init_detector(config_file, checkpoint_file, device='cuda:0')
img_list = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
print(f'检测到 {len(img_list)} 张图片')

class_names = model.dataset_meta['classes']

for img_path in img_list:
    img = mmcv.imread(img_path)
    result = inference_detector(model, img_path)
    pred_instances = result.pred_instances
    bboxes = pred_instances.bboxes.cpu().numpy()
    labels = pred_instances.labels.cpu().numpy()
    scores = pred_instances.scores.cpu().numpy()
    bboxes_with_scores = np.hstack([bboxes, scores[:, None]])

    # 可视化检测框
    mmcv.imshow_det_bboxes(
        img.copy(), bboxes_with_scores, labels, class_names=class_names,
        score_thr=0.3, show=False,
        out_file=os.path.join(output_dir, os.path.basename(img_path).replace('.', '_det.'))
    )

    # 可视化分割掩码（手动叠加）
    if hasattr(pred_instances, 'masks') and pred_instances.masks is not None:
        segms = pred_instances.masks.cpu().numpy()
        seg_img = img.copy()
        color_map = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0)]  # 你可以自定义颜色
        for idx, mask in enumerate(segms):
            color = color_map[labels[idx] % len(color_map)]
            mask = mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(seg_img, contours, -1, color, 2)
            seg_img[mask > 0] = seg_img[mask > 0] * 0.5 + np.array(color) * 0.5  # 半透明叠加
        mmcv.imwrite(seg_img, os.path.join(output_dir, os.path.basename(img_path).replace('.', '_seg.')))

print(f'可视化结果已保存到 {output_dir}')