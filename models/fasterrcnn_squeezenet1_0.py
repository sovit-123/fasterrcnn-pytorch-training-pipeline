"""
Faster RCNN model with the SqueezeNet1_0 model from Torchvision.
Torchvision link: https://pytorch.org/vision/stable/models.html#id15
Paper: https://arxiv.org/abs/1602.07360
"""

import torchvision

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

def create_model(num_classes=81, pretrained=False, coco_model=False):
    # Load the pretrained SqueezeNet1_0 backbone.
    backbone = torchvision.models.squeezenet1_0(weights='DEFAULT').features

    # We need the output channels of the last convolutional layers from
    # the features for the Faster RCNN model.
    # It is 512 for SqueezeNet1_0.
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
    model = create_model(81, pretrained=True, coco_model=True)
    summary(model)