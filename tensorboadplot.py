import json
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import os

def json_to_tensorboard(json_path, tb_dir, save_img_dir=None):
    train_loss = []
    val_acc = []
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                log = json.loads(line)
            except Exception:
                continue
            # 训练集loss
            if 'loss' in log and 'iter' in log:
                train_loss.append((log['iter'], log['loss']))
            # 验证集accuracy（以coco/bbox_mAP_50为例，可根据实际字段调整）
            if 'coco/bbox_mAP_50' in log and 'step' in log:
                val_acc.append((log['step'], log['coco/bbox_mAP_50']))

    writer = SummaryWriter(tb_dir)
    for step, loss in train_loss:
        writer.add_scalar('Train/Loss', loss, step)
    for step, acc in val_acc:
        writer.add_scalar('Val/Accuracy', acc, step)
    writer.close()
    print(f"写入TensorBoard日志到: {tb_dir}")

    # 保存图片
    if save_img_dir:
        os.makedirs(save_img_dir, exist_ok=True)
        # Loss 曲线
        if train_loss:
            steps, losses = zip(*train_loss)
            plt.figure()
            plt.plot(steps, losses, label='Train Loss')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title('Train Loss Curve')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(save_img_dir, 'train_loss.png'))
            plt.close()
        # Accuracy 曲线
        if val_acc:
            steps, accs = zip(*val_acc)
            plt.figure()
            plt.plot(steps, accs, label='Val Accuracy')
            plt.xlabel('Iteration')
            plt.ylabel('Accuracy')
            plt.title('Val Accuracy Curve')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(save_img_dir, 'val_accuracy.png'))
            plt.close()
        print(f"图片已保存到: {save_img_dir}")

# 用法示例
if __name__ == '__main__':
    json_to_tensorboard(
        'work_dirs/mask-rcnn_r50_fpn_1x_voc/final.json',
        'work_dirs/mask-rcnn_r50_fpn_1x_voc/tf_logs',
        save_img_dir='work_dirs/mask-rcnn_r50_fpn_1x_voc/curve_png'
    )