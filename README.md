
## DFINet: Dynamic feedback iterative network for infrared small target detection

**Paper**: [Dynamic feedback iterative network for infrared small target detection](https://www.sciencedirect.com/science/article/abs/pii/S0031320325006181)



## Framework of DFINet
<p align="center">
  <img src="Fig/framework.png" width="800">
</p>

## Visual
<p align="center">
  <img src="Fig/iter.png" width="800">
</p>

## 1.Training
To train DFINet from scratch, run:
```bash
python train.py
```

## 2.test
Training logs and checkpoints are available at:
- **百度网盘权重文件**: [权重文件](https://pan.baidu.com/s/1pgqABDSE4PlWjrvKZTKPhg)  
  **提取密码**: `ekts`
  
**This code is highly borrowed from [FANet](https://github.com/nikhilroxtomar/FANet). Thanks to Nikhil Kumar Tomar**。
**This code is highly borrowed from IRSTD-Toolbox. Thanks to Xinyi Ying.**

## Citation
```bash
@article{WU2026111958,
title = {DFINet: Dynamic feedback iterative network for infrared small target detection},
journal = {Pattern Recognition},
volume = {169},
pages = {111958},
year = {2026},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2025.111958},
url = {https://www.sciencedirect.com/science/article/pii/S0031320325006181},
author = {Jing Wu and Changhai Luo and Zhaobing Qiu and Liqiong Chen and Rixiang Ni and Yunxiang Li and Feng Huang and Jian Wu},
keywords = {Feedback iteration, Infrared small target detection, Feature mining, Error correction},
abstract = {Recently, deep learning-based methods have made impressive progress in infrared small target detection (IRSTD). However, the weak and variable nature of small targets constrains the feature extraction and scene adaptation of existing methods, leading to low data utilization and poor robustness. To address this issue, we innovatively introduce the feedback mechanism into IRSTD and propose the dynamic feedback iterative network (DFINet). The main motivation is to guide the model training and prediction utilizing the history prediction mask (HPMK) of previous rounds. On the one hand, in the training phase, DFINet can further mine the key features of real targets by training in multiple iterations with limited data; on the other hand, in the prediction phase, DFINet can correct the wrong results through feedback iterative to improve the model robustness. Specifically, we first propose the dynamic feedback feature fusion module (DFFFM), which dynamically interacts HPMK with feature maps through a hard attention mechanism to guide feature mining and error correction. Then, for better feature extraction, the cascaded hybrid pyramid pooling module (CHPP) is devised to capture both global and local information. Finally, we propose the dynamic semantic fusion module (DSFM), which innovatively utilizes feedback information to guide the fusion of high-level and low-level features for better feature representation in different scenarios. Extensive experimental results on publicly available datasets of NUDT-SIRST, IRSTD-1k, and SIRST Aug show that DFINet outperforms several state-of-the-art methods and achieves superior detection performance. Our code will be publicly available at https://github.com/uisdu/DFINet.}
}
```
