import os
import argparse
import json
from tensorboardX import SummaryWriter

def parse_log(log_path):
    train_loss = []
    val_acc = []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                log = json.loads(line)
            except Exception:
                continue
            # 训练loss：有'loss'且有'iter'，但没有'coco/'等字段
            if 'loss' in log and 'iter' in log and not any(k.startswith('coco/') for k in log):
                train_loss.append((log['iter'], log['loss']))
            # 验证acc：有'coco/bbox_mAP_50'或'coco/segm_mAP_50'，或其它验证集指标
            if any(k.startswith('coco/') for k in log):
                if 'coco/bbox_mAP_50' in log:
                    val_acc.append((log['step'], log['coco/bbox_mAP_50']))
                elif 'coco/segm_mAP_50' in log:
                    val_acc.append((log['step'], log['coco/segm_mAP_50']))
    return train_loss, val_acc

def log_to_tensorboard(log_path, tb_dir):
    train_loss, val_acc = parse_log(log_path)
    print(f"train_loss: {len(train_loss)}, val_acc: {len(val_acc)}")
    writer = SummaryWriter(tb_dir)
    for step, loss in train_loss:
        writer.add_scalar('Train/Loss', loss, step)
    for step, acc in val_acc:
        writer.add_scalar('Val/Accuracy', acc, step)
    writer.close()
    print(f"已写入TensorBoard日志到: {tb_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert MMDet log.json to TensorBoard event file')
    parser.add_argument('log_json', help='Path to MMDetection log.json')
    parser.add_argument('--out', default='tf_logs', help='Output TensorBoard log dir')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    log_to_tensorboard(args.log_json, args.out)