# A Simple Pipeline to Train PyTorch FasterRCNN Model

![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)



Train PyTorch FasterRCNN models easily on any custom dataset. Choose between official PyTorch models trained on COCO dataset, or choose any backbone from Torchvision classification models, or even write your own custom backbones. 

***You can run a Faster RCNN model with Mini Darknet backbone and Mini Detection Head at more than 150 FPS on an RTX 3080***.

![](readme_images/gif_1.gif)

## Get Started

​																								[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1oFxPpBeE8SzSQq7BTUv28IIqQeiHHLdj?usp=sharing) [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/sovitrath/custom-faster-rcnn-training-kaggle/notebook)

* [Find blog posts/tutorials on DebuggerCafe](#Tutorials)

## Updates

* **June 6 2025:** Support for both Pascal VOC and YOLO text file annotation type during training. Check [custom training section](#Train-on-Custom-Dataset)

* **August 28 2024:** SAHI image inference for all pretrained Torchvision Faster RCNN models integrated. [Find the script here](https://github.com/sovit-123/fasterrcnn-pytorch-training-pipeline/blob/main/sahi_inference.py).

* Filter classes to visualize during inference using the `--classes` command line argument with space separated class indices from the dataset YAML file. 

  For example, to visualize only persons in COCO dataset, use,  `python inference.py --classes 1 <rest of the command>`

  To visualize person and car, use, `python inference.py --classes 1 3 <rest of the command>`

* Added Deep SORT Real-Time tracking to `inference_video.py` and `onnx_video_inference.py`. Using `--track` command with the usual inference command. Support for **MobileNet** Re-ID for now.

## Custom Model Naming Conventions

***For this repository:***

* **Small head refers to 512 representation size in the Faster RCNN head and predictor.**
* **Tiny head refers to 256 representation size in the Faster RCNN head and predictor.**
* **Nano head refers to 128 representation size in the Faster RCNN head and predictor.**

## [Check All Available Model Flags](#A-List-of-All-Model-Flags-to-Use-With-the-Training-Script)

## Go To

* [Setup on Ubuntu](#Setup-for-Ubuntu)
* [Setup on Windows](#Setup-on-Windows)
* [Train on Custom Dataset](#Train-on-Custom-Dataset)
* [Inference](#Inference)
* [Evaluation](#Evaluation)
* [Available Models](#A-List-of-All-Model-Flags-to-Use-With-the-Training-Script)
* [Tutorials](#Tutorials)

## Setup on Ubuntu

1. Clone the repository.

   ```bash
   git clone https://github.com/sovit-123/fastercnn-pytorch-training-pipeline.git
   ```

   Optional: Initialize DINOv3 submodule for training DINOv3 Faster RCNN models. 

   ```bash
   git submodule update --init
   ```

2. Install requirements as per GPU.
   Install requirements on **RTX 30/40** (**Ampere and Ada Lovelace**) series and **T4/P100 GPUs**.

   ```bash
   pip install -r requirements.txt
   ```

**OR**	

Install requirements for **RTX 50** series and **Blackwell GPUs**. First install PyTorch >= 2.8 with CUDA >= 12.9

```
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129
```

Install rest of the requirements

```
pip install -r requirements_blackwell.txt
```

## Setup on Windows

1. **First you need to install Microsoft Visual Studio from [here](https://my.visualstudio.com/Downloads?q=Visual%20Studio%202017)**. Sing In/Sing Up by clicking on **[this link](https://my.visualstudio.com/Downloads?q=Visual%20Studio%202017)** and download the **Visual Studio Community 2017** edition.

   ![](readme_images/vs-2017-annotated.jpg)

   Install with all the default chosen settings. It should be around 6 GB. Mainly, we need the C++ Build Tools.

2. Then install the proper **`pycocotools`** for Windows.

   ```bash
   pip install git+https://github.com/gautamchitnis/cocoapi.git@cocodataset-master#subdirectory=PythonAPI
   ```

3. Clone the repository.

   ```bash
   git clone https://github.com/sovit-123/fastercnn-pytorch-training-pipeline.git
   ```

4. Then install the remaining **[requirements](https://github.com/sovit-123/pytorch-efficientdet-api/blob/main/requirements.txt)** except for `pycocotools`.

   Install requirements on **RTX 30/40** (**Ampere and Ada Lovelace**) series and **T4/P100 GPUs**.

   ```bash
   pip install -r requirements.txt
   ```

**OR**	

Install requirements for **RTX 50** series and **Blackwell GPUs**. First install PyTorch >= 2.8 with CUDA >= 12.9

```
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129
```

Install rest of the requirements (apart from `pycocotools`)

```
pip install -r requirements_blackwell.txt
```

## Using Custom Weights

Some models like **DINOv3 based Faster RCNN** models require the pretrained weights to be present locally. Put all the DINOv3 backbone weights in the `weights` directory. The respective model files will load them from the directory.

You can download the weights by [filling the form here](https://github.com/facebookresearch/dinov3/tree/main?tab=readme-ov-file#pretrained-models).

For example, for the FasterRCNN DINOv3 ConvNext Tiny model (`models/fasterrcnn_dinov3_convnext_tiny.py`) the weights are loaded using the following syntax with relative path.

```python
# Relative to parent fasterrcnn directory.
REPO_DIR = 'dinov3'
# Relative to parent fasterrcnn directory or the absolute path.
WEIGHTS_URL = 'weights/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth'

self.backbone = torch.hub.load(
    REPO_DIR, 
    "dinov3_convnext_tiny", 
    source='local', 
    weights=WEIGHTS_URL
)
```

## Train on Custom Dataset

Taking an exmaple of the [smoke dataset](https://www.kaggle.com/didiruh/smoke-pascal-voc) from Kaggle. Let's say that the dataset is in the `data/smoke_pascal_voc` directory in the following format. And the `smoke.yaml` is in the `data_configs` directory. Assuming, we store the smoke data in the `data` directory

```bash
├── data
│   ├── smoke_pascal_voc
│   │   ├── archive
│   │   │   ├── train
│   │   │   └── valid
│   └── README.md
├── data_configs
│   └── smoke.yaml
├── models
│   ├── create_fasterrcnn_model.py
│   ...
│   └── __init__.py
├── outputs
│   ├── inference
│   └── training
│       ...
├── readme_images
│   ...
├── torch_utils
│   ├── coco_eval.py
│   ...
├── utils
│   ├── annotations.py
│   ...
├── datasets.py
├── inference.py
├── inference_video.py
├── __init__.py
├── README.md
├── requirements.txt
└── train.py
```

The content of the `smoke.yaml` should be the following. The folder containing the annotation files can either point to **Pascal VOC XML** files or **YOLO text labels** folder. The images and labels (for both Pascal VOC XML and YOLO text files) can be either in the same folder or in different folders because the image and annotation files are matched based on the file names during dataset preparation. 

**If the data config file (shown below) points to Pascal VOC XML annotations, the `CLASSES` field can contain the class names in any order after the `__background__` class. If the data config file points to YOLO text file annotation folder, the `CLASSES` should contain the class names in the order as present in the YOLO dataset `data.yaml` file. This is necessary to maintain indexing order during training.**

![](readme_images/fasterrcnn_yolo_config_example.png)

```yaml
# Images and labels direcotry should be relative to train.py
TRAIN_DIR_IMAGES: ../../xml_od_data/smoke_pascal_voc/archive/train/images
TRAIN_DIR_LABELS: ../../xml_od_data/smoke_pascal_voc/archive/train/annotations # This can contain .xml or .txt files
# VALID_DIR should be relative to train.py
VALID_DIR_IMAGES: ../../xml_od_data/smoke_pascal_voc/archive/valid/images
VALID_DIR_LABELS: ../../xml_od_data/smoke_pascal_voc/archive/valid/annotations # This can contain .xml or .txt files

# Class names.
CLASSES: [
    '__background__',
    'smoke'
]

# Number of classes (object classes + 1 for background class in Faster RCNN).
NC: 2

# Whether to save the predictions of the validation set while training.
SAVE_VALID_PREDICTION_IMAGES: True
```

***Note that*** *the data and annotations can be in the same directory as well. In that case, the TRAIN_DIR_IMAGES and TRAIN_DIR_LABELS will save the same path. Similarly for VALID images and labels. The `datasets.py` will take care of that*.

Next, to start the training, you can use the following command.

**Command format:**

During training, we need to provide a `--label-type` argument which should be either `yolo` or `pascal_voc` depending on the annotation folder path in the data configuration file above. Default is `pascal_voc`

```bash
python train.py --data <path to the data config YAML file> --epochs 100 --model <model name (defaults to fasterrcnn_resnet50)> --name <folder name inside output/training/> --batch 16 --label-type <pascal_voc or yolo>
```

**In this case, the exact command would be:**

```bash
python train.py --data data_configs/smoke.yaml --epochs 100 --model fasterrcnn_resnet50_fpn --name smoke_training --batch 16 --label-type pascal_voc
```

**The terimal output should be similar to the following:**

```
Number of training samples: 665
Number of validation samples: 72

3,191,405 total parameters.
3,191,405 training parameters.
Epoch     0: adjusting learning rate of group 0 to 1.0000e-03.
Epoch: [0]  [ 0/84]  eta: 0:02:17  lr: 0.000013  loss: 1.6518 (1.6518)  time: 1.6422  data: 0.2176  max mem: 1525
Epoch: [0]  [83/84]  eta: 0:00:00  lr: 0.001000  loss: 1.6540 (1.8020)  time: 0.0769  data: 0.0077  max mem: 1548
Epoch: [0] Total time: 0:00:08 (0.0984 s / it)
creating index...
index created!
Test:  [0/9]  eta: 0:00:02  model_time: 0.0928 (0.0928)  evaluator_time: 0.0245 (0.0245)  time: 0.2972  data: 0.1534  max mem: 1548
Test:  [8/9]  eta: 0:00:00  model_time: 0.0318 (0.0933)  evaluator_time: 0.0237 (0.0238)  time: 0.1652  data: 0.0239  max mem: 1548
Test: Total time: 0:00:01 (0.1691 s / it)
Averaged stats: model_time: 0.0318 (0.0933)  evaluator_time: 0.0237 (0.0238)
Accumulating evaluation results...
DONE (t=0.03s).
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.001
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.002
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.001
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.009
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.007
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.029
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.074
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.028
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.088
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.167
SAVING PLOTS COMPLETE...
...
Epoch: [4]  [ 0/84]  eta: 0:00:20  lr: 0.001000  loss: 0.9575 (0.9575)  time: 0.2461  data: 0.1662  max mem: 1548
Epoch: [4]  [83/84]  eta: 0:00:00  lr: 0.001000  loss: 1.1325 (1.1624)  time: 0.0762  data: 0.0078  max mem: 1548
Epoch: [4] Total time: 0:00:06 (0.0801 s / it)
creating index...
index created!
Test:  [0/9]  eta: 0:00:02  model_time: 0.0369 (0.0369)  evaluator_time: 0.0237 (0.0237)  time: 0.2494  data: 0.1581  max mem: 1548
Test:  [8/9]  eta: 0:00:00  model_time: 0.0323 (0.0330)  evaluator_time: 0.0226 (0.0227)  time: 0.1076  data: 0.0271  max mem: 1548
Test: Total time: 0:00:01 (0.1116 s / it)
Averaged stats: model_time: 0.0323 (0.0330)  evaluator_time: 0.0226 (0.0227)
Accumulating evaluation results...
DONE (t=0.03s).
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.137
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.313
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.118
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.029
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.175
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.428
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.204
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.306
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.347
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.140
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.424
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.683
SAVING PLOTS COMPLETE...
```

## Distributed Training

**Training on 2 GPUs**:

```bash
export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.launch --nproc_per_node=2 --use_env train.py --data data_configs/smoke.yaml --epochs 100 --model fasterrcnn_resnet50_fpn --name smoke_training --batch 16
```

## Inference

### Image Inference on COCO Pretrained Model

By default using **Faster RCNN ResNet50 FPN V2** model.

```bash
python inference.py
```

Use model of your choice with an image input.

```bash
python inference.py --model fasterrcnn_mobilenetv3_large_fpn --input example_test_data/image_1.jpg
```

### Image Inference in Custom Trained Model

In this case you only need to give the weights file path and input file path. The config file and the model name are optional. If not provided they will will be automatically inferred from the weights file.

```bash
python inference.py --input data/inference_data/image_1.jpg --weights outputs/training/smoke_training/last_model_state.pth
```

### Video Inference on COCO Pretrrained Model

```bash
python inference_video.py
```

### Video Inference in Custom Trained Model

```bash
python inference_video.py --input data/inference_data/video_1.mp4 --weights outputs/training/smoke_training/last_model_state.pth 
```

### Tracking using COCO Pretrained Models

```bash
# Track all COCO classes (Faster RCNN ResNet50 FPN V2).
python inference_video.py --track --model fasterrcnn_resnet50_fpn_v2 --show

# Track all COCO classes (Faster RCNN ResNet50 FPN V2) using own video.
python inference_video.py --track --model fasterrcnn_resnet50_fpn_v2 --show --input ../inference_data/video_1.mp4

# Tracking only person class (index 1 in COCO pretrained). Check `COCO_91_CLASSES` attribute in `data_configs/coco.yaml` for more information.
python inference_video.py --track --model fasterrcnn_resnet50_fpn_v2 --show --input ../inference_data/video_4.mp4 --classes 1

# Tracking only person and car classes (indices 1 and 3 in COCO pretrained). Check `COCO_91_CLASSES` attribute in `data_configs/coco.yaml` for more information.
python inference_video.py --track --model fasterrcnn_resnet50_fpn_v2 --show --input ../inference_data/video_4.mp4 --classes 1 3

# Tracking using custom trained weights. Just provide the path to the weights instead of model name.
python inference_video.py --track --weights outputs/training/fish_det/best_model.pth --show --input ../inference_data/video_6.mp4
```

## Evaluation

Replace the required arguments according to your need.

```bash
python eval.py --model fasterrcnn_resnet50_fpn_v2 --weights outputs/training/trial/best_model.pth --data data_configs/aquarium.yaml --batch 4
```

You can use the following command to show a table for **class-wise Average Precision** (`--verbose` additionally needed).

```bash
python eval.py --model fasterrcnn_resnet50_fpn_v2 --weights outputs/training/trial/best_model.pth --data data_configs/aquarium.yaml --batch 4 --verbose
```

## A List of All Model Flags to Use With the Training Script

The following command expects the `coco` dataset to be present one directory back inside the `input` folder in XML format. You can find the dataset [here on Kaggle](https://www.kaggle.com/datasets/sovitrath/coco-xml-format). Check the `data_configs/coco.yaml` for more details. You can change the relative dataset path in the YAML file according to your structure.

```bash
# Usage 
python train.py --model fasterrcnn_resnet50_fpn_v2 --data data_configs/coco.yaml
```

**OR USE ANY ONE OF THE FOLLOWING**

```
[
    'fasterrcnn_convnext_small',
    'fasterrcnn_convnext_tiny',
    'fasterrcnn_custom_resnet', 
    'fasterrcnn_darknet',
    'fasterrcnn_efficientnet_b0',
    'fasterrcnn_efficientnet_b4',
    'fasterrcnn_mbv3_small_nano_head',
    'fasterrcnn_mbv3_large',
    'fasterrcnn_mini_darknet_nano_head',
    'fasterrcnn_mini_darknet',
    'fasterrcnn_mini_squeezenet1_1_small_head',
    'fasterrcnn_mini_squeezenet1_1_tiny_head',
    'fasterrcnn_mobilenetv3_large_320_fpn', # Torchvision COCO pretrained
    'fasterrcnn_mobilenetv3_large_fpn', # Torchvision COCO pretrained
    'fasterrcnn_nano',
    'fasterrcnn_resnet18',
    'fasterrcnn_resnet50_fpn_v2', # Torchvision COCO pretrained
    'fasterrcnn_resnet50_fpn',  # Torchvision COCO pretrained
    'fasterrcnn_resnet101',
    'fasterrcnn_resnet152',
    'fasterrcnn_squeezenet1_0',
    'fasterrcnn_squeezenet1_1_small_head',
    'fasterrcnn_squeezenet1_1',
    'fasterrcnn_vitdet',
    'fasterrcnn_vitdet_tiny',
    'fasterrcnn_mobilevit_xxs',
    'fasterrcnn_regnet_y_400mf'
]
```

## Tutorials

* [Wheat Detection using Faster RCNN and PyTorch](https://debuggercafe.com/wheat-detection-using-faster-rcnn-and-pytorch/)
* [Plant Disease Detection using the PlantDoc Dataset and PyTorch Faster RCNN](https://debuggercafe.com/plant-disease-detection-using-plantdoc/)
* [Small Scale Traffic Light Detection using PyTorch](https://debuggercafe.com/small-scale-traffic-light-detection/)
