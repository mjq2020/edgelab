# 基于 OpenMMLab 框架的多任务网络模型库

[English](./README.md) | 简体中文

## 简介

这是一个基于 OpenMMLab 框架开发的一个适用于多种视觉任务主干网络的模型训练库。

利用 OpenMMLab 框架，我们可以轻松地开发一个新的主干网络，并利用 MMClassification，MMDetection 和 MMPose 来在分类、检测等任务上进行基准测试。

## 配置环境

运行环境需要 [PyTorch](https://pytorch.org/get-started/locally/) 和以下 OpenMMLab 仓库:

- [MIM](https://github.com/open-mmlab/mim): 一个用于管理 OpenMMLab 包和实验的命令行工具
- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab 计算机视觉基础库
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab 图像分类工具箱和基准测试。除了分类任务，它同时用于提供多样的主干网络
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab 检测工具箱和基准测试
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab 检测工具箱和基准测试

如果只是使用CPU进行训练，则通过以下命令即可配置完成相关环境。

```bash
pip install -r ./requirements/requirements.txt    
mim install mmcv-full                       #todo
```

- 对于上述命令若安装过程过慢可使用国内镜像，例如使用清华镜像可在后面添加`-i https://pypi.tuna.tsinghua.edu.cn/simple` 加快安装速度。

如果需要使用GPU进行训练，可查看GPU训练相关环境[配置教程](./docs/zh_cn/get_started.md)

<details>
<summary>Windows</summary>

将本仓库克隆至本地后进入本项目文件夹，同时将本项目文件夹路径添加至环境变量中，变量名为PYTHONPATH，添加完成后可执行以下命令查看是否添加成功。

```bash
set PYTHONPATH
```

若显示本项目地址路径表明添加成功。
</details>

<details>
<summary>Linux</summary>

同样需要将本项目的文件路径添加至系统环境变量中，变量名为PYTHONPATH，其可通过修改~/.bashrc 文件以保证在后续新终端中可用。
在终端中依次执行以下命令即可：

```bash
echo export PYTHONPATH=`pwd`:\$PYTHONPATH >> ~/.bashrc
source ~/.bashrc
```

</details>

## 数据准备

建议将数据集放置在与本项目同级目录下，其数据与本项目结构组织如下所示：\

```text
edgelab/
├── ...
│   .
│   .
datasets/
├── imagenet
│   ├── train
│   ├── val
│   └── meta
│       ├── train.txt
│       └── val.txt
└── coco
    ├── annotations
    │   ├── instance_train2017.json
    │   └── instance_val2017.json
    ├── train2017
    └── val2017
```

这里，我们只列举了用于训练和验证 ImageNet（分类任务）、和 COCO（检测任务）的必要文件。

- **提示：** 数据集文件可根据自己习惯放置，只需要在数据集的注释文件中能够找到对应的数据文件即可。

## 开始使用

1. 首先需要确定所做的任务类型，属于目标检测、分类、或回归；确定后可根据需要选择需要的模型，并确定模型的配置文件。
2. 这里我们以YOLOv3人体检测模型为例演示如何修改配置文件中的相关参数训练自己的数据集，以及训练和导出。

### 1.修改配置文件中的数据集路径

1. 在[configs](./configs)文件夹下寻找所需要修改的[配置文件](./configs/yolo/yolov3_192_node2_person.py)。
2. 在配置文件中找到变量`data_root`，将变量值替换为自己所用数据集的根目录的路径。
3. 检查配合文件中的`img_perfix`和`ann_file`路径是否正确。
    - **提示：** `img_perfix` 为数据集图片的路径，`ann_file` 为数据集注释文件的路径

### 2.训练

1.在配置好的环境终端中执行以下命令即可开始训练人体检测模型。

- **提示：** 虚拟换环境需要启动对应的虚拟环境

```shell
mim train mmdet $CONFIG_PATH --workdir=$WORKERDIR --gpus=1
```

#### 参数解释：

- `$CONFIG_PATH`需替换为你所使用的模型配置文件本示例中为[yolov3_192_node2_person.py](./configs/yolo/yolov3_192_node2_person.py)的路径。
- `$WORKERDIR`为训练过程产生的日志文件和权重文件保存的文件夹名称，默认为`worke_dir`
- `--gpus=1`表示使用一块GPU训练，若使用CPU进行训练可设置参数为`--gpus=0`

2.在模型训练完成后会在`$WORKERDIR`文件夹下产生相应的日志文件和模型权重文件(后缀为`.pth`)。

**提示：** 对于更多训练任务的使用可查看更多[训练示例](./docs/zh_cn/train_example.md)

### 3.导出ONNX

在模型训练完成后可将pth文件导出到onnx文件格式，并通过onnx转为其他想要使用的格式。
假设此时环境在本项目路径下，可通过执行如下命令导出刚才训练的人体检测模型至onnx格式。

```shell
#导出onnx时不对模型进行量化
python ./tools/torch2onnx.py  --config ./config/yolo/yolov3_192_node2_person.py --checkpoint work_dirs/yolov3_192_node2_person/latest.pth  --imgsz $IMGSZ 

#导出onnx时同时对模型进行量化(PTQ)
python ./tools/torch2onnx.py --config ./config/yolo/yolov3_192_node2_person.py --checkpoint work_dirs/yolov3_192_node2_person/latest.pth --imgsz $IMGSZ --quantize
```

##### 参数解释:

- `--config`:模型训练相关配置文件的路径。
- `--checkpoint`: 训练完成后在相应文件夹下产生的模型权重文件路径(`.pth`后缀)。
- `$OUTPUT`:导出onnx的文件名，其路径在`$MODEL_PATH`下，与原模型同路径。
- `$IMGSZ`:模型所输入数据大小，图片数据则为宽高，音频数据则为长度。

## 相关教程

1. 对于导出除ONNX格式外的其他格式可参考教程[文档](./docs/zh_cn/tutorials)
2. 如何在[colab]()中使用[aroboflow](https://app.roboflow.com/)数据集训练可参见[相关教程](./docs/zh_cn/tutorials/)
3. [模型量化](./docs/zh_cn/tutorials/quantize.md)
4. [更多相关数据集结构](./docs/zh_cn/tutorials/datasets_config.md)
5. [相关工具的使用](./docs/zh_cn/tutorials/use_tools.md)
6. 更多关于[环境配置](./docs/zh_cn/tutorials/env_config.md)

## 模型库

对于mmdetection、mmpos、mmclassify所支持的模型均可使用。
本项目直接实现的算法有以下：\
[PFLD](https://arxiv.org/pdf/1902.10859.pdf)\
[EAT](https://arxiv.org/pdf/2204.11479.pdf)

## FAQ

对于在环境配置与训练过程中可能出现的问题可先查看[相关问题解决文档](./docs/zh_cn/faq.md)
查看。若没能解决您的问题可提出[issue](https://github.com/Seeed-Studio/edgelab/issues)，
我们会尽快为您解决。

## 许可证

edgelab 目前以 Apache 2.0 的许可证发布，但是其中有一部分功能并不是使用的 Apache2.0 许可证，我们在[许可证](./LICENSES.md)
中详细地列出了这些功能以及他们对应的许可证，如果您正在从事盈利性活动，请谨慎参考此文档。




