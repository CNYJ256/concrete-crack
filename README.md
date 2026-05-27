# 混凝土裂缝检测
使用Unet、deeplabv3plus、fcn8s模型进行混凝土裂缝检测，提供训练、评估和预测功能。
## 安装依赖
```bash
pip install -r requirements.txt
```
## 快速开始
### 训练模型
```bash
D:/Anaconda_envs/envs/torch/python.exe D:/repos/concrete-crack/run_train.py
```
### 模型预测
```bash
python run_predict.py --model unet --checkpoint <path> --input <path>
```

## 论文参考
1. Liu, Z., Cao, Y., Wang, Y., & Wang, W. (2019). Computer vision-based concrete crack detection using U-net fully convolutional networks. Automation in Construction, 104, 129-139. https://doi.org/10.1016/j.autcon.2019.04.005
2. Liu, Y., Yao, J., Lu, X., Xie, R., & Li, L. (2019). DeepCrack: A deep hierarchical feature learning architecture for crack segmentation. Neurocomputing, 338, 139-153. https://doi.org/10.1016/j.neucom.2019.01.036
3. Unet: Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention (pp. 234-241). Springer, Cham.
4. DeepLabv3+: Chen, L. C., Zhu, Y., Papandreou, G., Schroff, F., & Adam, H. (2018). Encoder-decoder with atrous separable convolution for semantic image segmentation. In Proceedings of the European conference on computer vision (ECCV) (pp. 801-818).
5. FCN8s: Long, J., Shelhamer, E., & Darrell, T. (2015). Fully convolutional networks for semantic segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 3431-3440).

## 数据来源
https：//github.com/yhlleo/DeepCrack

### 数据库信息
数据库总共537张RGB彩色图像
所有图像都有像素级标注的分割图
图像尺寸：544×384像素
通道数：RGB彩色图像（3通道）
标注格式：二值掩码（黑白图像）
数据集已划分为训练集300张+测试集237张，建议直接使用原始划分，以便与原论文指标对比

## 感谢
感谢指导老师的支持和帮助，以及数据集提供者的贡献，使得本项目得以顺利进行。

## 许可证
本项目采用MIT许可证，允许任何人自由使用、修改和分发代码，但需保留原作者的版权声明和许可信息。