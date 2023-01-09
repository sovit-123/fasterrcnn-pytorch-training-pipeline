"""
Faster RCNN model with the ResNet18 backbone from Torchvision.
Torchvision link: https://pytorch.org/vision/stable/models.html#id10
ResNet paper: https://arxiv.org/pdf/1512.03385.pdf
"""

import torchvision
import torch.nn as nn

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

def create_model(num_classes, pretrained=True, coco_model=False):
    # Load the pretrained ResNet18 backbone.
    model_backbone = torchvision.models.resnet18(weights='DEFAULT')

    conv1 = model_backbone.conv1
    bn1 = model_backbone.bn1
    relu = model_backbone.relu
    max_pool = model_backbone.maxpool
    layer1 = model_backbone.layer1
    layer2 = model_backbone.layer2
    layer3 = model_backbone.layer3
    layer4 = model_backbone.layer4

    backbone = nn.Sequential(
        conv1, bn1, relu, max_pool, 
        layer1, layer2, layer3, layer4
    )

    # We need the output channels of the last convolutional layers from
    # the features for the Faster RCNN model.
    # It is 512 for ResNet18.
    backbone.out_channels = 512

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

if __name__ == '__main__':
    from model_summary import summary
    model = create_model(num_classes=81, pretrained=True, coco_model=True)
    summary(model)