import os
import glob
import re
import json
from datetime import datetime
import matplotlib.pyplot as plt

log_dir = 'work_dirs'
curve_dir = 'curve_dir'
os.makedirs(curve_dir, exist_ok=True)
exp_pattern = re.compile(r'mask-rcnn_r50_fpn_1x_voc.*')
date_threshold = datetime.strptime('2024-05-20', '%Y-%m-%d')

def parse_log(log_path):
    train_iters, train_loss = [], []
    val_iters, val_loss, val_acc = [], [], []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                log = json.loads(line)
            except Exception:
                continue
            if log.get('mode') == 'train' and 'loss' in log:
                train_iters.append(log['iter'])
                train_loss.append(log['loss'])
            if log.get('mode') == 'val':
                if 'loss' in log:
                    val_iters.append(log['iter'])
                    val_loss.append(log['loss'])
                if 'accuracy' in log:
                    val_acc.append((log['iter'], log['accuracy']))
                if 'bbox_mAP_50' in log:
                    val_acc.append((log['iter'], log['bbox_mAP_50']))
                if 'mask_mAP_50' in log:
                    val_acc.append((log['iter'], log['mask_mAP_50']))
    return train_iters, train_loss, val_iters, val_loss, val_acc

# 递归查找所有子目录下的 .json 日志文件
exp_folders = [os.path.join(log_dir, d) for d in os.listdir(log_dir) if exp_pattern.match(d)]
log_files = []
for exp in exp_folders:
    for root, dirs, files in os.walk(exp):
        for file in files:
            if file.endswith('.json'):
                log_path = os.path.join(root, file)
                mtime = datetime.fromtimestamp(os.path.getmtime(log_path))
                if mtime >= date_threshold:
                    log_files.append(log_path)

if not log_files:
    print('未找到5/20日以后的日志文件。')
    exit(0)

for log_path in log_files:
    print(f'处理日志: {log_path}')
    train_iters, train_loss, val_iters, val_loss, val_acc = parse_log(log_path)
    plt.figure(figsize=(10, 6))
    has_curve = False
    if train_iters and train_loss:
        plt.plot(train_iters, train_loss, label='Train Loss')
        has_curve = True
    if val_iters and val_loss:
        plt.plot(val_iters, val_loss, label='Val Loss')
        has_curve = True
    if val_acc:
        acc_iters, acc_vals = zip(*val_acc)
        plt.plot(acc_iters, acc_vals, label='Val Accuracy/mAP')
        has_curve = True
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.title(f'Training/Validation Loss & Accuracy\n{os.path.basename(log_path)}')
    if has_curve:
        plt.legend()
    plt.grid(True)
    out_png = os.path.join(curve_dir, os.path.basename(log_path).replace('.json', '_curve.png'))
    plt.savefig(out_png)
    print(f'保存曲线图到: {out_png}')
    plt.close()