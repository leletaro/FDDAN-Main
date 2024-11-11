from pathlib import Path
from ultralytics import SAM, YOLO
import torch
import cv2
import numpy as np
import time
from collections import defaultdict

# 定义图像数据路径
img_data_path = 'MaskRDD-140'

# 定义检测模型和SAM模型的路径
det_model_path = "/PATH"   # File source：DBBG Train
sam_model_path = "/PATH"   # File source：https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2_l.pt

# 根据CUDA是否可用选择设备
device = '0' if torch.cuda.is_available() else 'cpu'

# 定义输出目录，默认为None
output_dir = None

# 初始化检测模型和SAM模型
det_model = YOLO(det_model_path)
sam_model = SAM(sam_model_path)

# 获取图像数据路径
data = Path(img_data_path)

# 如果输出目录未定义，则生成默认的输出目录
if not output_dir:
    output_dir = data.parent / f"{data.stem}_FDDAN"
    Path(output_dir).mkdir(exist_ok=True, parents=True)

# 解析多边形格式的YOLO标注
def parse_polygon_labels(label_path, img_width, img_height):
    polygons = []
    with open(label_path, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            cls_id = int(values[0])
            coords = np.array(values[1:]).reshape(-1, 2)
            abs_coords = (coords * [img_width, img_height]).astype(int)
            polygons.append((cls_id, abs_coords))
    return polygons

# 对图像数据进行检测
det_results = det_model(data, stream=True, device=device)

# 统计IoU和分割推理时间
iou_scores_by_class = defaultdict(list)
inference_times = []

# 遍历检测结果
for result in det_results:
    start_time = time.time()
    
    orig_img = cv2.imread(result.path)
    orig_img_height, orig_img_width = orig_img.shape[:2]
    
    # 加载YOLO格式标注
    yolo_label_path = str(Path(result.path).with_suffix('.txt'))
    ground_truth_polygons = parse_polygon_labels(yolo_label_path, orig_img_width, orig_img_height)
    
    # 检测的边框和类别
    boxes = result.boxes.xyxy
    if len(boxes) == 0:
        continue  # 跳过没有检测结果的情况

    # 获取分割结果
    sam_results = sam_model(result.orig_img, bboxes=boxes, verbose=False, save=False, device=device)
    segments = sam_results[0].masks.xyn
    
    for i, s in enumerate(segments):
        if len(s) == 0:
            print(f"IoU for segment {i} with ground truth: No segmentation output")
            continue  # 跳过没有分割结果的情况

        abs_coords = (s * [orig_img_width, orig_img_height]).astype(int)
        mask = np.zeros((orig_img_height, orig_img_width), dtype=np.uint8)
        cv2.fillPoly(mask, [abs_coords], 255)

        # 计算每个标注的IoU
        for cls_id, gt_coords in ground_truth_polygons:
            gt_mask = np.zeros((orig_img_height, orig_img_width), dtype=np.uint8)
            cv2.fillPoly(gt_mask, [gt_coords], 255)

            # 计算IoU
            intersection = np.logical_and(mask, gt_mask).sum()
            union = np.logical_or(mask, gt_mask).sum()
            iou = intersection / union if union != 0 else 0

            print(f"IoU for segment {i} with ground truth of class {cls_id}: {iou}")
            if iou > 0:
                iou_scores_by_class[cls_id].append(iou)

    # 记录分割推理时间
    end_time = time.time()
    inference_times.append(end_time - start_time)

# 计算每个类别的平均IoU和平均推理时间（仅统计有分割结果的情况）
for cls_id, ious in iou_scores_by_class.items():
    avg_iou = np.mean(ious) if ious else 0
    print(f"Average IoU for class {cls_id} (non-zero segments only): {avg_iou}")

average_inference_time = np.mean(inference_times) if inference_times else 0
print(f"Average segmentation inference time per image: {average_inference_time} seconds")

