# FDDAN-Main
# Synergistic Enhancement: A Study on the Design of Large Models Assisted by End-to-End Road Damage Prompt Network and Methods for Quantification of Damage Morphological Features

## Overview

This repository implements **FDDAN**, a novel prompt-assisted network designed for pixel-wise segmentation and quantification of road damage. Road maintenance requires precise damage segmentation and accurate morphological quantification, but traditional methods struggle with the diversity of damage types and the high cost of annotation. **FDDAN** addresses these challenges by integrating a large model with a specialized prompt-assisted network for zero-shot pixel-level segmentation and morphological quantification.

Paper:Synergistic Enhancement:A Study on theDesign of Large Models Assisted by End-to-EndRoad Damage Prompt Network andMethods for Quantification of DamageMorphological Features
/n
DOI: 10.1109/JSEN.2025.3569295

## Key Features

- **End-to-End Pipeline**: Combines pre-processing and post-processing stages for streamlined road damage segmentation and quantification.
- **Dynamic Snake Convolution & Deformable Attention**: Utilized in the pre-processing stage to focus on irregularly shaped damage areas.
- **Dy-Detect Head**: Enhances the model’s ability to detect complex damage types through modules for scale, spatial, and task awareness.
- **Quantification with Normal Vector Estimation and Branch-Point Detection**: Improves accuracy in quantifying key morphological parameters of damage.

## Architecture

The FDDAN model architecture includes two main stages:

### 1. Pre-Processing (Segmentation)

An end-to-end Damage Box Bounding Prompt Network (DBBPN) is introduced in the pre-processing stage, focusing on irregularly shaped damage areas. Key components include:

- **Damage Bounding Box Generator (DBBG)**: Identifies the location and size of road damage using a detection network, typically from the YOLO series.
- **Bounding Box Encoder (BBE)**: Decodes bounding boxes into prompt information for the large model, guiding accurate segmentation of detected damage.

### 2. Post-Processing (Quantification)

A damage quantification method based on **normal vector estimation** and **branch-point detection** enhances the accuracy of quantifying damage features. This stage processes segmentation results for final quantification output.

## Evaluation

The model was evaluated on the **RDD2022 dataset** and a self-made validation dataset, **MaskRDD-140**. Results show that FDDAN achieves superior performance over existing models, measured by **F1 score**, **mAP50**, and **MIoU** metrics. 

## Challenges and Solutions

Two primary challenges are addressed:

1. **Precise Segmentation of Road Damage**: Handled through the DBBPN, which provides prompt information to the large model.
2. **Quantification of Morphological Feature Parameters**: Solved by the quantification method in the post-processing stage, leveraging ROI binary superposition with normal vector estimation and branch-point detection.

## Installation

To get started, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/your_username/FDDAN-main.git
cd FDDAN-main
pip install -r requirements.txt
```

## Usage

1. **Data Preparation**: Prepare your dataset according to the format specified in the `data/` folder.
2. **Training**: To train the model on your data, run:
   ```bash
   python train.py --config configs/config.yaml
   ```
3. **Evaluation**: To evaluate the model, run:
   ```bash
   python evaluate.py --dataset RDD2022 --metrics f1,map50,miou
   ```

## Results

The evaluation on RDD2022 and MaskRDD-140 datasets shows that FDDAN outperforms other methods in achieving high segmentation and quantification accuracy for diverse types of road damage.

## License

This project is licensed under the MIT License.

## Acknowledgments

This research benefits from the robust road damage detection and quantification methodologies developed in the field of computer vision and machine learning.
