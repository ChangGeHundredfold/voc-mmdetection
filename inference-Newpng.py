from mmdet.apis import DetInferencer

inferencer = DetInferencer(
    weights='work_dirs/sparse-rcnn_r50_fpn_1x_voc/epoch_20.pth',
    device='cuda:0'
)

# 推理并保存可视化结果
inferencer('demo/out', out_dir='outputs/out/sparse_rcnn', no_save_pred=False)