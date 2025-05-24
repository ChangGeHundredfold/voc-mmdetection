from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import os

event_path = 'work_dirs/mask-rcnn_r50_fpn_1x_voc/tf_logs'  # 你的event文件目录
ea = event_accumulator.EventAccumulator(event_path)
ea.Reload()

# 获取所有scalar标签
tags = ea.Tags()['scalars']
os.makedirs('curve_png', exist_ok=True)
for tag in tags:
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    vals = [e.value for e in events]
    plt.figure()
    plt.plot(steps, vals)
    plt.title(tag)
    plt.xlabel('step')
    plt.ylabel(tag)
    plt.savefig(f'curve_png/{tag.replace("/", "_")}.png')
    plt.close()
print('所有曲线已保存到 curve_png 文件夹')