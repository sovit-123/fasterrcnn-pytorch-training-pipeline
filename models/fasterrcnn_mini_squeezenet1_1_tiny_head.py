"""
Backbone: SqueezeNet1_1 with changed backbone features. Had to tweak a few 
input and output features in the backbone for this.

Torchvision link: https://pytorch.org/vision/stable/models.html#id15
SqueezeNet repo: https://github.com/forresti/SqueezeNet/tree/master/SqueezeNet_v1.1

Detection Head: Custom Tiny Faster RCNN Head with only 256 representation size. 
"""

import torchvision
import torch
import torch.nn.functional as F

from torch import nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models

    Args:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super().__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x

class FastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Args:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            torch._assert(
                list(x.shape[2:]) == [1, 1],
                f"x has the wrong shape, expecting the last two dimensions to be [1,1] instead of {list(x.shape[2:])}",
            )
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas

def create_model(num_classes=81, pretrained=True, coco_model=True):
    # Load the pretrained SqueezeNet1_1 backbone.
    backbone = torchvision.models.squeezenet1_1(pretrained=pretrained).features

    # Change the number of features in backbone[12] block to reduce model size.
    # Although the weights for this block may become random, 
    # we still have the previous layers with ImageNet weights. So, 
    # will still perform pretty well in transfer learning.
    backbone[12].squeeze = nn.Conv2d(512, 32, kernel_size=(1, 1), stride=(1, 1))
    backbone[12].expand1x1 = nn.Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1))
    backbone[12].expand3x3 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    # We need the output channels of the last convolutional layers from
    # the features for the Faster RCNN model.
    # It is 128 for this custom SqueezeNet1_1.
    backbone.out_channels = 128

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

    representation_size = 256

    # Box head.
    box_head = TwoMLPHead(
        in_channels=backbone.out_channels * roi_pooler.output_size[0] ** 2, 
        representation_size=representation_size
    )

    # Box predictor.
    box_predictor = FastRCNNPredictor(representation_size, num_classes)

    # Final Faster RCNN model.
    model = FasterRCNN(
        backbone=backbone,
        num_classes=None, # Num classes shoule be None when `box_predictor` is provided.
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        box_head=box_head,
        box_predictor=box_predictor
    )
    return model

if __name__ == '__main__':
    from model_summary import summary
    model = create_model(num_classes=81, pretrained=True, coco_model=True)
    summary(model)