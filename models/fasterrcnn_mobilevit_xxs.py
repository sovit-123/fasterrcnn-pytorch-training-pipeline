"""
Faster RCNN Head with MobileViT XXS (Extra Extra Small) as backbone.
You need to install vision_transformers library for this.
Find the GitHub project here:
https://github.com/sovit-123/vision_transformers
"""

import torchvision
import torch.nn as nn
import sys

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

try:
    from vision_transformers.models.mobile_vit import mobilevit_xxs
except:
    print('Please intall Vision Transformers to use MobileViT backbones')
    print('You can do pip install vision_transformers')
    print('Or visit the following link for the latest updates')
    print('https://github.com/sovit-123/vision_transformers')
    assert ('vision_transformers' in sys.modules), 'vision_transformers not found'

def create_model(num_classes, pretrained=True, coco_model=False):
    # Load the backbone.
    model_backbone = mobilevit_xxs(pretrained=pretrained)

    backbone = nn.Sequential(*list(model_backbone.children())[:-1])

    # Output channels from the final convolutional layer.
    backbone.out_channels = 320

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
    try:
        summary(model)
    except:
        print(model)
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")