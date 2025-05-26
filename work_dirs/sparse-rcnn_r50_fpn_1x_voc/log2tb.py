# import os
# import json
# from tensorboardX import SummaryWriter

# base_dir = r'd:\DATA\51\mid_pj\voc-mmdetection-main\work_dirs\sparse-rcnn_r50_fpn_1x_voc'

# # 自动查找所有含有vis_data/scalars.json的子目录
# exp_dirs = []
# for root, dirs, files in os.walk(base_dir):
#     if 'scalars.json' in files:
#         exp_dirs.append(root)

# for exp_dir in exp_dirs:
#     writer = SummaryWriter(log_dir=exp_dir)
#     scalar_file = os.path.join(exp_dir, 'scalars.json')
#     with open(scalar_file, 'r') as f:
#         for step, line in enumerate(f):
#             data = json.loads(line)
#             for k, v in data.items():
#                 writer.add_scalar(k, v, step)
#     writer.close()

# import os
# import json
# from tensorboardX import SummaryWriter

# base_dir = r'd:\DATA\51\mid_pj\voc-mmdetection-main\work_dirs\sparse-rcnn_r50_fpn_1x_voc'

# # 查找所有含有scalars.json的vis_data目录，并按文件夹名排序
# exp_dirs = []
# for root, dirs, files in os.walk(base_dir):
#     if 'scalars.json' in files:
#         exp_dirs.append(root)
# exp_dirs.sort()  # 按文件夹名排序，确保顺序

# writer = SummaryWriter(log_dir=base_dir + '_tb_log')
# global_step = 0

# for exp_dir in exp_dirs:
#     scalar_file = os.path.join(exp_dir, 'scalars.json')
#     with open(scalar_file, 'r') as f:
#         for line in f:
#             data = json.loads(line)
#             for k, v in data.items():
#                 writer.add_scalar(k, v, global_step)
#             global_step += 1

# writer.close()
# print(f"TensorBoard log saved to: {base_dir + '_tb_log'}")

import os
import json
from tensorboardX import SummaryWriter

base_dir = r'D:\DATA\51\mid_pj\voc-mmdetection-main\work_dirs\sparse-rcnn_r50_fpn_1x_voc'

# 查找所有含有scalars.json的目录，并按文件夹名排序
json_files = []
for root, dirs, files in os.walk(base_dir):
    if 'scalars.json' in files:
        json_files.append(os.path.join(root, 'scalars.json'))
json_files.sort()  # 按路径排序，确保顺序

writer = SummaryWriter(log_dir=base_dir + '_tb_log')
global_step = 0

for json_file in json_files:
    with open(json_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            for k, v in data.items():
                writer.add_scalar(k, v, global_step)
            global_step += 1

writer.close()
print(f"TensorBoard log saved to: {base_dir + '_tb_log'}")