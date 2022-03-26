# Line as a Visual Sentence with LineTR

This repository contains the inference code, pretrained model, and demo scripts of the following [paper](https://arxiv.org/abs/2109.04753). It supports both point(SuperPoint) and line features(LSD+LineTR).
```
@ARTICLE{syoon-2021-linetr,
  author={Sungho Yoon and Ayoung Kim},
  journal={IEEE Robotics and Automation Letters}, 
  title={Line as a Visual Sentence: Context-Aware Line Descriptor for Visual Localization}, 
  year={2021},
  volume={6},
  number={4},
  pages={8726-8733},
  doi={10.1109/LRA.2021.3111760}}
```
<p align="center">
  <a href="https://arxiv.org/abs/2109.04753"><img src="assets/linetr-github.gif" width="80%"/></a>
</p>

---------------------------------------------------
## Announcements
- Training codes are available! I hope you like it.
- PL-Loc with LineTR is on [visual localization benchmark (InLoc dataset)](https://www.visuallocalization.net/benchmark/).

## TODO List
- [x] Support training codes for self-supervised homography augmentation
- [ ] Support a different line detector

## Abstract

Along with feature points for image matching, line features provide additional constraints to solve visual geometric problems in robotics and computer vision (CV). Although recent convolutional neural network (CNN)-based line descriptors are promising for viewpoint changes or dynamic environments, we claim that the CNN architecture has innate disadvantages to abstract variable line length into the fixed-dimensional descriptor. In this paper, we effectively introduce the Line-Transformer dealing with variable lines.  Inspired by natural language processing (NLP) tasks where sentences can be understood and abstracted well in neural nets, we view a line segment as a sentence that contains points (words). By attending to well-describable points on a line dynamically, our descriptor performs excellently on variable line length. We also propose line signature networks sharing the line's geometric attributes to neighborhoods. Performing as group descriptors, the networks enhance line descriptors by understanding lines' relative geometries. Finally, we present the proposed line descriptor and matching in a Point and Line Localization (PL-Loc). We show that the visual localization with feature points can be improved using our line features. We validate the proposed method for homography estimation and visual localization.

## Getting Started
This code was tested with Python 3.6 and PyTorch 1.8 on Ubuntu 18.04.
```
# create and activate a new conda environment
conda create -y --name linetr
conda activate linetr

# install the dependencies
conda install -y python=3.6
pip install -r requirements.txt
```

## Command
There are two demo scripts:
1. `demo_LineTR.py` : run a live demo on a camera or video file
2. `match_line_pairs.py` : find line correspondence for image pairs, listed in input_pairs.txt

Keyboard control:
* `n`: select the current frame as the anchor
* `e`/`r`: increase/decrease the keypoint confidence threshold
* `d`/`f`: increase/decrease the nearest neighbor matching threshold for **keypoints**
* `c`/`v`: increase/decrease the nearest neighbor matching threshold for **keylines**
* `k`: toggle the visualization of keypoints
* `q`: quit

The scripts are partially reusing [SuperGluePretrainedNetwork](https://github.com/magicleap/SuperGluePretrainedNetwork).

## How to train LineTR
The training scripts and configurations can be modified depending on the development environment. 
Please also refer to the config files in 'dataloaders/confs/homography.yaml' and 'train_manager.yaml' to adjust #gpu, batch size, nWorkers, etc. 

The raw images should be located in 'assets/dataset/raw_images/' and their dataset files will be saved at 'assets/dataset/dataset_h5/'.

There are three steps for training:
```
python dataloaders/build_homography_dataset.py  # 1. building a homography dataset
python tools/divide_train_val_test.py           # 2. divding files into training and validation sets
python train.py                                 # 3. training process
```

## BibTeX Citation

```
@ARTICLE{syoon-2021-linetr,
  author={Sungho Yoon and Ayoung Kim},
  journal={IEEE Robotics and Automation Letters}, 
  title={Line as a Visual Sentence: Context-Aware Line Descriptor for Visual Localization}, 
  year={2021},
  volume={6},
  number={4},
  pages={8726-8733},
  doi={10.1109/LRA.2021.3111760}}
```

## Acknowledgment

This work was fully supported by [Localization in changing city] project funded by NAVER LABS Corporation.
