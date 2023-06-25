# WUDA
WUDA

## Citation 
Please cite this paper in your publications if it helps your research:
```
@INPROCEEDINGS{10094958,
  author={Liu, Shengjie and Zhu, Chuang and Li, Yuan and Tang, Wenqi},
  booktitle={ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={WUDA: Unsupervised Domain Adaptation Based on Weak Source Domain Labels}, 
  year={2023},
  volume={},
  number={},
  pages={1-5},
  doi={10.1109/ICASSP49357.2023.10094958}}
```
## Introduction

The acquisition of semantic labels is often time-consuming, many scenarios only have weak labels (e.g. bounding boxes). When weak supervision and cross-domain problems coexist, this paper defines a new task: unsupervised domain adaptation based on weak source domain labels (WUDA).
<div align="center">
<img src="https://github.com/bupt-ai-cz/WUDA/assets/33684330/778d2140-f183-4fe6-b63d-92bd31a4fda3" height="350">
</div>


## Frameworks
### WSSS-UDA
We first propose the framework WSSS-UDA: in order to take advantage of the fine box labels in the source domain, first perform box-supervised segmentation in the source domain, then the problem is transformed into a UDA task.
<div align="center">
<img src="https://github.com/bupt-ai-cz/WUDA/assets/33684330/c116de26-a5d1-4e0e-83c9-3b3b1a0a3c17" height="300">
</div>

#### Step 1: Box Supervised Segmentic Segmentation

Clone this repo.
```
git clone https://github.com/bupt-ai-cz/WUDA.git
```

Perform GrabCut on the source domain images to obtain pseudo-labels that can be used for supervised semantic segmentation training.

```
cd grabcut
grabcut.py --txt_path path/to/bboxes --image_path path/to/images --save_path path/to/pseudo_labels
```

Train a semantic segmentation model using the pseudo-labels obtained from GrabCut. This stage corresponds to the warm-up phase of the unsupervised domain adaptation algorithm. We use [HIAST](https://github.com/bupt-ai-cz/HIAST) in our work.
#### Setp 2: Unsupervised Domain Adaptation For Semantic Segmentation

This step corresponds to the adversarial training or self-training (or both) stage in unsupervised domain adaptation for semantic segmentation algorithms. Please refer to the second stage (adversarial training) and the third stage (self-training) of [HIAST](https://github.com/bupt-ai-cz/HIAST) for more details.

We also benchmark other UDA algorithms within framework WSSS-UDA: [CBST](https://github.com/yzou2/CBST), [CPSL](https://github.com/lslrh/cpsl).

### TDOD-WSSS

we also implement the cross-domain process on the object detection task and propose the framework TDOD-WSSS (Figure 4(b)): First, use the bounding box labels of the source domain to train the object detection model, then predict bounding boxes on the target domain. Finally, implement box-supervised segmentation in the target domain.

<div align="center">
<img src="https://github.com/bupt-ai-cz/WUDA/assets/33684330/4fab737b-6866-467e-aee6-2906298f61bd" height="300">
</div>

#### Step 1: Target Domain Object Detection

In this step, it is required to train an object detection model using the source domain data and provide predictions on the target domain images.

We benchmark [Yolov5](https://github.com/ultralytics/yolov5) within this framework.

#### Step 2: Box Supervised Segmentic Segmentation

Please refer to Step 2 in the framework WSSS-UDA.

## Main Results
<div align="center">
<img src="https://github.com/bupt-ai-cz/WUDA/assets/33684330/9d9faef7-69fd-4058-9db6-df0867a93cbf" height="400">
</div>

<div align="center">
<img src="https://github.com/bupt-ai-cz/WUDA/assets/33684330/19c12d2c-6bc6-408a-bb60-a274f4fc9f79" height="240">
</div>

<div align="center">
<img src="https://github.com/bupt-ai-cz/WUDA/assets/33684330/a35bb07c-e21c-44a5-bc32-da2ede27fd11" height="150">
</div>


## Representation Shift and Constructed Dataset

### Representation Shift
<div align="center">
<img src="https://github.com/bupt-ai-cz/WUDA/assets/33684330/6f661144-7c47-4e35-b547-e2c0d21999e1" height="300">
</div>


### Constructed Dataset
We construct a series of datasets with different domain shifts and further analyze the impact of multiple domain shifts on the two frameworks.
<div align="center">
<img src="https://github.com/bupt-ai-cz/WUDA/assets/33684330/60755742-35a2-4ba8-906c-68978cbfc5e9" height="280">
</div>

