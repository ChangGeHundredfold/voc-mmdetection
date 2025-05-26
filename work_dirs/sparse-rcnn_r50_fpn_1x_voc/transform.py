import json
from torch.utils.tensorboard import SummaryWriter
import os

def convert_json_to_tensorboard(json_file_path, log_dir='runs/experiment_1'):
    """
    Converts a JSON file containing training and evaluation logs to TensorBoard format.

    Args:
        json_file_path (str): Path to the input JSON file (e.g., 'all.json').
        log_dir (str): Directory to save TensorBoard log files.
    """
    # Ensure the log directory exists
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    print(f"Processing file: {json_file_path}")
    print(f"TensorBoard logs will be saved in: {log_dir}")

    try:
        with open(json_file_path, 'r') as f:
            for line_num, line in enumerate(f):
                try:
                    log_entry = json.loads(line.strip())

                    # Check if it's a training log (contains "iter" and "loss")
                    if "iter" in log_entry and "loss" in log_entry and "step" in log_entry:
                        step = log_entry["step"]  # Global iteration count

                        # Log general training metrics
                        if "lr" in log_entry:
                            writer.add_scalar('Train/Learning_Rate', log_entry["lr"], step)
                        writer.add_scalar('Train/Total_Loss', log_entry["loss"], step)
                        if "grad_norm" in log_entry:
                            writer.add_scalar('Train/Gradient_Norm', log_entry["grad_norm"], step)
                        if "data_time" in log_entry: # Data loading time per iteration
                            writer.add_scalar('Train/Data_Time_Iter', log_entry["data_time"], step)
                        if "time" in log_entry: # Total time per iteration
                            writer.add_scalar('Train/Iteration_Time', log_entry["time"], step)
                        if "memory" in log_entry:
                             writer.add_scalar('Train/Memory_Usage_MB', log_entry["memory"], step)

                        # Log detailed metrics for each stage (s0 to s5)
                        for i in range(6):
                            prefix = f"s{i}."
                            stage_name = f"Train/Stage_{i}"
                            if f"{prefix}loss_cls" in log_entry:
                                writer.add_scalar(f'{stage_name}/Loss_Cls', log_entry[f"{prefix}loss_cls"], step)
                            if f"{prefix}pos_acc" in log_entry:
                                writer.add_scalar(f'{stage_name}/Pos_Acc', log_entry[f"{prefix}pos_acc"], step)
                            if f"{prefix}loss_bbox" in log_entry:
                                writer.add_scalar(f'{stage_name}/Loss_Bbox', log_entry[f"{prefix}loss_bbox"], step)
                            if f"{prefix}loss_iou" in log_entry:
                                writer.add_scalar(f'{stage_name}/Loss_IoU', log_entry[f"{prefix}loss_iou"], step)

                    # Check if it's an evaluation log (contains "pascal_voc/mAP")
                    elif "pascal_voc/mAP" in log_entry and "step" in log_entry:
                        epoch_step = log_entry["step"] # Epoch number

                        writer.add_scalar('Eval/Pascal_VOC_mAP', log_entry["pascal_voc/mAP"], epoch_step)
                        if "pascal_voc/AP50" in log_entry:
                            writer.add_scalar('Eval/Pascal_VOC_AP50', log_entry["pascal_voc/AP50"], epoch_step)
                        if "data_time" in log_entry: # Data loading time for evaluation
                            writer.add_scalar('Eval/Data_Time_Epoch', log_entry["data_time"], epoch_step)
                        if "time" in log_entry: # Total time for evaluation
                            writer.add_scalar('Eval/Evaluation_Time_Epoch', log_entry["time"], epoch_step)
                    # else:
                        # Optional: print a message for lines that don't match known formats
                        # print(f"Skipping unrecognized log line {line_num + 1}: {line.strip()[:100]}...")

                except json.JSONDecodeError:
                    print(f"Warning: JSON parsing error on line {line_num + 1}, skipping: {line.strip()}")
                except KeyError as e:
                    print(f"Warning: Missing key {e} on line {line_num + 1}, skipping: {log_entry}")

    except FileNotFoundError:
        print(f"Error: File {json_file_path} not found.")
        return
    except Exception as e:
        print(f"An error occurred while processing the file: {e}")
        return
    finally:
        writer.close()
        print("Processing complete.")

if __name__ == '__main__':
    # --- Configuration ---
    # 🔹 Make sure 'all.json' is in the same directory as this script,
    #    or provide the full path.
    json_input_file = 'all.json'

    # 🔹 You can change this to your preferred directory for TensorBoard logs.
    tensorboard_output_directory = 'runs/my_training_run'
    # --- End Configuration ---

    # Check for PyTorch and TensorBoard installation
    try:
        import torch
        from torch.utils.tensorboard import SummaryWriter
        print(f"PyTorch version: {torch.__version__}")
        # This check ensures tensorboard is available as part of torch.utils
    except ImportError:
        print("Error: PyTorch or TensorBoard is not installed. Please install them:")
        print("  pip install torch torchvision torchaudio")
        print("  pip install tensorboard")
        exit()

    convert_json_to_tensorboard(json_input_file, tensorboard_output_directory)

    print(f"\n🚀 To view results in TensorBoard, run the following command in your terminal:")
    print(f"   tensorboard --logdir={tensorboard_output_directory}")
    print(f"Then open the provided URL (usually http://localhost:6006/) in your browser.")