"""
Faster RCNN model with the ResNet50 backbone from Torchvision.
Torchvision link: https://pytorch.org/vision/stable/models.html#id10
ResNet paper: https://arxiv.org/pdf/1512.03385.pdf
"""

import torchvision
import torch.nn as nn

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

def create_model(num_classes, pretrained=True, coco_model=False):
    # Load the pretrained ResNet50 backbone.
    conv1 = torchvision.models.resnet50(pretrained=True).conv1
    bn1 = torchvision.models.resnet50(pretrained=True).bn1
    relu = torchvision.models.resnet50(pretrained=True).relu
    max_pool = torchvision.models.resnet50(pretrained=True).maxpool
    layer1 = torchvision.models.resnet50(pretrained=True).layer1
    layer2 = torchvision.models.resnet50(pretrained=True).layer2
    layer3 = torchvision.models.resnet50(pretrained=True).layer3
    layer4 = torchvision.models.resnet50(pretrained=True).layer4

    backbone = nn.Sequential(
        conv1, bn1, relu, max_pool, 
        layer1, layer2, layer3, layer4
    )

    # We need the output channels of the last convolutional layers from
    # the features for the Faster RCNN model.
    # It is 2048 for ResNet50.
    backbone.out_channels = 2048

    # Generate anchors using the RPN. Here, we are using 5x3 anchors.
    # Meaning, anchors with 5 different sizes and 3 different aspect 
    # ratios.
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    # Feature maps to perform RoI cropping.
    # If backbone returns a Tensor, `featmap_names` is expected to
    # be [0]. We can choose which feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )

    # Final Faster RCNN model.
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )
    return model