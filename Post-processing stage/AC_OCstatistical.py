from pathlib import Path
from ultralytics import SAM, YOLO
import torch
import cv2  # OpenCV库用于图像处理
import numpy as np

# 定义图像数据路径
img_data_path = 'AC_OCstatisticalanalysis'

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
    output_dir = data.parent / f"{data.stem}_statistic"
    # 创建输出目录
    Path(output_dir).mkdir(exist_ok=True, parents=True)

# 初始化类别2和3的像素统计
alligator_crack_pixels = 0
other_corruption_pixels = 0

# 对图像数据进行检测
det_results = det_model(data, stream=True, device=device)

# 遍历检测结果
for result in det_results:
    # 获取类别ID
    class_ids = result.boxes.cls.int().tolist()  # noqa
    # 如果有检测到物体
    if len(class_ids):
        # 获取检测框坐标
        boxes = result.boxes.xyxy  # Boxes object for bbox outputs
        # 使用SAM模型进行分割
        sam_results = sam_model(result.orig_img, bboxes=boxes, verbose=False, save=False, device=device)
        # 获取分割结果
        segments = sam_results[0].masks.xyn  # noqa
        
        # 读取原始图像以进行可视化处理
        orig_img = cv2.imread(result.path)
        orig_img_height, orig_img_width = orig_img.shape[:2]
        
        # 遍历每个分割区域
        for i in range(len(segments)):
            s = segments[i]
            # 如果分割区域为空，则跳过
            if len(s) == 0:
                continue
            
            # 将相对坐标转换为绝对像素坐标
            abs_coords = (s * np.array([orig_img_width, orig_img_height])).astype(int)

            # 创建一个空的mask图像
            mask = np.zeros((orig_img_height, orig_img_width), dtype=np.uint8)

            # 填充多边形区域
            cv2.fillPoly(mask, [abs_coords], 255)

            # 判断类别并统计像素
            if class_ids[i] == 2:  # Alligator Crack
                alligator_crack_pixels += np.sum(mask == 255)
            elif class_ids[i] == 3:  # Other Corruption
                other_corruption_pixels += np.sum(mask == 255)

            # 可视化分割结果（可选）
            color_mask = np.zeros_like(orig_img)
            color = np.random.randint(0, 255, (3,), dtype=np.uint8)  # 随机颜色
            color_mask[mask == 255] = color
            visualized_img = cv2.addWeighted(orig_img, 0.7, color_mask, 0.3, 0)

# 输出统计结果
print(f"类别 2 (Alligator Crack) 的像素总数: {alligator_crack_pixels}")
print(f"类别 3 (Other Corruption) 的像素总数: {other_corruption_pixels}")
