"""
Faster RCNN model with the MobileNetV3 backbone.

Reference: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
"""

import torchvision

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

def create_model(num_classes):
    # Load the pretrained MobileNetV3 large features.
    backbone = torchvision.models.mobilenet_v3_large(pretrained=True).features

    # We need the output channels of the last convolutional layers from
    # the features for the Faster RCNN model.
    # It is 960 for MobilNetV3.
    backbone.out_channels = 960

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